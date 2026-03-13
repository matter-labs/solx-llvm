//===- EVMUtil.cpp - EVM specific MLIR utilities --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EVM specific MLIR utilities implementation.
//
// TODO:
// Why does via-ir generate a signed (instead of an unsigned) comparison
// (usually involving offsets) in some of the abi encoding/decoding codegen?
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SolToStandard/EVMUtil.h"
#include "mlir/Conversion/SolToStandard/EVMConstants.h"
#include "mlir/Conversion/SolToStandard/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"

using namespace mlir;

unsigned evm::getAlignment(evm::AddrSpace addrSpace) {
  // FIXME: Confirm this!
  return addrSpace == evm::AddrSpace_Stack ? evm::ByteLen_Field
                                           : evm::ByteLen_Byte;
}

unsigned evm::getAlignment(Value ptr) {
  auto ty = cast<LLVM::LLVMPointerType>(ptr.getType());
  return getAlignment(static_cast<evm::AddrSpace>(ty.getAddressSpace()));
}

unsigned evm::getCallDataHeadSize(Type ty) {
  if (isa<IntegerType>(ty) || isa<sol::EnumType>(ty) ||
      isa<sol::BytesType>(ty) || sol::hasDynamicallySizedElt(ty) ||
      sol::isAddressLikeType(ty))
    return 32;

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
    return arrTy.getSize() * getCallDataHeadSize(arrTy.getEltType());

  llvm_unreachable("NYI: Other types");
}

int64_t evm::getMallocSize(Type ty) {
  // String type is dynamic.
  assert(!isa<sol::StringType>(ty));
  // Array type.
  if (auto arrayTy = dyn_cast<sol::ArrayType>(ty)) {
    assert(!arrayTy.isDynSized());
    return arrayTy.getSize() * 32;
  }
  // Struct type.
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    // FIXME: Is the memoryHeadSize 32 for all the types (assuming padding is
    // enabled by default) in StructType::memoryDataSize?
    return structTy.getMemberTypes().size() * 32;
  }

  // Value type.
  return 32;
}

unsigned evm::getStorageSlotCount(Type ty) {
  if (isa<IntegerType>(ty) || isa<sol::EnumType>(ty) ||
      isa<sol::BytesType>(ty) || isa<sol::MappingType>(ty) ||
      isa<sol::FuncRefType>(ty) || sol::hasDynamicallySizedElt(ty) ||
      sol::isAddressLikeType(ty))
    return 1;

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
    return arrTy.getSize() * getStorageSlotCount(arrTy.getEltType());

  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    int64_t sum = 0;
    for (Type memTy : structTy.getMemberTypes())
      sum += getStorageSlotCount(memTy);
    return sum;
  }

  llvm_unreachable("NYI: Other types");
}

bool evm::canBePacked(Type ty) {
  // Scalars can be packed within a slot.
  if (isa<IntegerType>(ty) || isa<sol::EnumType>(ty) ||
      isa<sol::BytesType>(ty) || isa<sol::FuncRefType>(ty) ||
      sol::isAddressLikeType(ty))
    return true;

  // Aggregates are slot-aligned and cannot be packed.
  if (isa<sol::ArrayType>(ty) || isa<sol::StructType>(ty) ||
      isa<sol::MappingType>(ty) || isa<sol::StringType>(ty))
    return false;

  llvm_unreachable("NYI");
}

unsigned evm::getStorageByteSize(Type ty) {
  assert(canBePacked(ty) && "Only packable types have byte size");

  if (auto intTy = dyn_cast<IntegerType>(ty))
    // Bool occupies 1 byte in storage.
    return intTy.getWidth() == 1 ? 1 : intTy.getWidth() / 8;

  if (auto bytesTy = dyn_cast<sol::BytesType>(ty))
    return bytesTy.getSize();

  // Enums can have at most 256 members, so always 1 byte.
  if (isa<sol::EnumType>(ty))
    return 1;

  // Address-like types are 20 bytes.
  if (sol::isAddressLikeType(ty))
    return 20;

  // Internal function reference.
  if (isa<sol::FuncRefType>(ty))
    return 8;

  llvm_unreachable("NYI");
}

Value evm::Builder::normalizeABIScalarForEncoding(
    Type ty, Value val, Location loc,
    std::optional<sol::DataLocation> srcDataLoc) {
  mlir::solgen::BuilderExt bExt(b, loc);
  bool fromCalldata = srcDataLoc && *srcDataLoc == sol::DataLocation::CallData;

  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() == 256)
      return val;

    assert(intTy.getWidth() < 256 &&
           "Expected integer types no wider than 256 bits");
    auto valTy = cast<IntegerType>(val.getType());
    assert((valTy.getWidth() == intTy.getWidth() || valTy.getWidth() == 256) &&
           "Expected integer value with source width or i256 width");

    // If the value is already at source width, only widen to i256 for storing.
    if (valTy.getWidth() == intTy.getWidth())
      return bExt.genIntCast(/*width=*/256, intTy.isSigned(), val, loc);

    // For bool, use 'x != 0' cleanup, otherwise regular integer cast.
    Value trunc = bExt.genIntCastWithBoolCleanup(intTy.getWidth(),
                                                 intTy.isSigned(), val, loc);

    // Finally, extend to 256 bits.
    Value normalized =
        bExt.genIntCast(/*width=*/256, intTy.isSigned(), trunc, loc);
    if (fromCalldata) {
      Value revertCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 val, normalized);
      genRevert(revertCond, loc);
    }
    return normalized;
  }

  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    Value normalized =
        bExt.genIntCast(/*width=*/256, /*isSigned=*/false, val, loc);
    Value revertCond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, normalized,
                                bExt.genI256Const(enumTy.getMax()));
    if (fromCalldata)
      genRevert(revertCond, loc);
    else
      genPanic(mlir::evm::PanicCode::EnumConversionError, revertCond, loc);
    return normalized;
  }

  if (sol::isAddressLikeType(ty)) {
    Value casted = bExt.genIntCast(/*width=*/256, /*isSigned=*/false, val, loc);
    APInt mask = APInt::getLowBitsSet(256, 160);
    Value normalized =
        b.create<arith::AndIOp>(loc, casted, bExt.genI256Const(mask));
    if (fromCalldata) {
      Value revertCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 casted, normalized);
      genRevert(revertCond, loc);
    }
    return normalized;
  }

  if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
    Value casted = bExt.genIntCast(/*width=*/256, /*isSigned=*/false, val, loc);
    if (bytesTy.getSize() == 32)
      return casted;

    assert(bytesTy.getSize() < 32 && "Expected fixed-bytes width <= 32");
    APInt mask = APInt::getHighBitsSet(256, bytesTy.getSize() * 8);
    Value normalized =
        b.create<arith::AndIOp>(loc, casted, bExt.genI256Const(mask));
    if (fromCalldata) {
      Value revertCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 casted, normalized);
      genRevert(revertCond, loc);
    }
    return normalized;
  }

  return val;
}

Value evm::Builder::genHeapPtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto heapAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), evm::AddrSpace_Heap);
  return b.create<LLVM::IntToPtrOp>(loc, heapAddrSpacePtrTy, addr);
}

Value evm::Builder::genCallDataPtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto callDataAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), evm::AddrSpace_CallData);
  return b.create<LLVM::IntToPtrOp>(loc, callDataAddrSpacePtrTy, addr);
}

Value evm::Builder::genReturnDataPtr(Value addr,
                                     std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto callDataAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), evm::AddrSpace_ReturnData);
  return b.create<LLVM::IntToPtrOp>(loc, callDataAddrSpacePtrTy, addr);
}

Value evm::Builder::genStoragePtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto storageAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), evm::AddrSpace_Storage);
  return b.create<LLVM::IntToPtrOp>(loc, storageAddrSpacePtrTy, addr);
}

Value evm::Builder::genTStoragePtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto tstorageAddrSpacePtrTy = LLVM::LLVMPointerType::get(
      b.getContext(), evm::AddrSpace_TransientStorage);
  return b.create<LLVM::IntToPtrOp>(loc, tstorageAddrSpacePtrTy, addr);
}

Value evm::Builder::genCodePtr(Value addr, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto storageAddrSpacePtrTy =
      LLVM::LLVMPointerType::get(b.getContext(), evm::AddrSpace_Code);
  return b.create<LLVM::IntToPtrOp>(loc, storageAddrSpacePtrTy, addr);
}

Value evm::Builder::genFreePtr(std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  return b.create<yul::MLoadOp>(loc, bExt.genI256Const(64));
}

void evm::Builder::genFreePtrUpd(Value freePtr, Value size,
                                 std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // FIXME: Shouldn't we check for overflow in the freePtr + size operation
  // and generate PanicCode::ResourceError?
  //
  // FIXME: Do we need round up the size to a multiple of 32 here?
  Value newFreePtr = b.create<arith::AddIOp>(loc, freePtr, size);

  // Generate the PanicCode::ResourceError check.
  //
  // TODO: Do we need to imposes a hard limit of ``type(uint64).max`` here?
  b.create<yul::MStoreOp>(loc, bExt.genI256Const(64), newFreePtr);
}

Value evm::Builder::genMemAlloc(Value size, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value freePtr = genFreePtr(loc);
  genFreePtrUpd(freePtr, size, loc);
  return freePtr;
}

Value evm::Builder::genMemAlloc(AllocSize size,
                                std::optional<Location> locArg) {
  assert(size % 32 == 0);
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  return genMemAlloc(bExt.genI256Const(size), loc);
}

Value evm::Builder::genMemAllocForDynArray(Value sizeVar, Value sizeInBytes,
                                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  mlir::solgen::BuilderExt bExt(b, loc);

  // dynSize is size + length-slot where length-slot's size is 32 bytes.
  auto dynSizeInBytes =
      b.create<arith::AddIOp>(loc, sizeInBytes, bExt.genI256Const(32));
  auto memPtr = genMemAlloc(dynSizeInBytes, loc);
  b.create<yul::MStoreOp>(loc, memPtr, sizeVar);
  return memPtr;
}

/// Generates the memory allocation and optionally the zero initializer code.
Value evm::Builder::genMemAlloc(Type ty, bool zeroInit, ValueRange initVals,
                                Value sizeVar, int64_t recDepth,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  recDepth++;

  // Array type.
  if (auto arrayTy = dyn_cast<sol::ArrayType>(ty)) {
    Value memPtr;
    assert(arrayTy.getDataLocation() == sol::DataLocation::Memory);

    Value sizeInBytes, dataPtr;
    // FIXME: Round up size for byte arays.
    if (arrayTy.isDynSized()) {
      // Dynamic allocation is only performed for the outermost dimension.
      if (sizeVar && recDepth == 0) {
        sizeInBytes =
            b.create<arith::MulIOp>(loc, sizeVar, bExt.genI256Const(32));
        memPtr = genMemAllocForDynArray(sizeVar, sizeInBytes, loc);
        dataPtr = b.create<arith::AddIOp>(loc, memPtr, bExt.genI256Const(32));
      } else {
        return bExt.genI256Const(mlir::evm::MemoryLayout::zeroPointer);
      }
    } else {
      sizeInBytes = bExt.genI256Const(evm::getMallocSize(ty));
      memPtr = genMemAlloc(sizeInBytes, loc);
      dataPtr = memPtr;
    }
    assert(sizeInBytes && dataPtr && memPtr);

    Type eltTy = arrayTy.getEltType();

    // Multi-dimensional array / array of structs.
    if (isa<sol::StructType>(eltTy) || isa<sol::ArrayType>(eltTy)) {
      if (!initVals.empty()) {
        // This is probably a multi-dimensional array literal op. The inner
        // allocation should be done by another array literal op. So we only
        // store the offsets.
        Value addr = dataPtr;
        for (auto val : initVals) {
          b.create<yul::MStoreOp>(loc, addr, val);
          addr = b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
        }
        return memPtr;
      }

      //
      // Store the offsets to the "inner" allocations.
      //
      // Generate the loop for the stores of offsets.

      // `size` should be a multiple of 32.
      b.create<scf::ForOp>(
          loc, /*lowerBound=*/bExt.genIdxConst(0),
          /*upperBound=*/bExt.genCastToIdx(sizeInBytes),
          /*step=*/bExt.genIdxConst(32), /*initArgs=*/ValueRange{},
          /*builder=*/
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
            Value incrMemPtr = b.create<arith::AddIOp>(
                loc, dataPtr, bExt.genCastToI256(indVar));
            b.create<yul::MStoreOp>(
                loc, incrMemPtr,
                genMemAlloc(eltTy, zeroInit, initVals, sizeVar, recDepth, loc));
            b.create<scf::YieldOp>(loc);
          });

    } else if (zeroInit) {
      Value callDataSz = b.create<yul::CallDataSizeOp>(loc);
      b.create<yul::CallDataCopyOp>(loc, dataPtr, callDataSz, sizeInBytes);

    } else {
      Value addr = dataPtr;
      for (auto val : initVals) {
        b.create<yul::MStoreOp>(loc, addr, val);
        addr = b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
      }
    }

    return memPtr;
  }

  // String type.
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    if (sizeVar)
      return genMemAllocForDynArray(
          sizeVar, bExt.genRoundUpToMultiple<32>(sizeVar), loc);
    return bExt.genI256Const(mlir::evm::MemoryLayout::zeroPointer);
  }

  // Struct type.
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    Value memPtr = genMemAlloc(evm::getMallocSize(ty), loc);
    assert(structTy.getDataLocation() == sol::DataLocation::Memory);

    for (auto memTy : structTy.getMemberTypes()) {
      Value initVal;
      if (isa<sol::StructType>(memTy) || isa<sol::ArrayType>(memTy)) {
        initVal = genMemAlloc(memTy, zeroInit, {}, sizeVar, recDepth, loc);
        b.create<yul::MStoreOp>(loc, memPtr, initVal);
      } else if (zeroInit) {
        b.create<yul::MStoreOp>(loc, memPtr, bExt.genI256Const(0));
      }
    }
    return memPtr;
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genMemAlloc(Type ty, bool zeroInit, ValueRange initVals,
                                Value sizeVar, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (sizeVar)
    sizeVar = bExt.genIntCast(256, false, sizeVar);

  return genMemAlloc(ty, zeroInit, initVals, sizeVar,
                     /*recDepth=*/-1, loc);
}

Value evm::Builder::genDynSize(Value addr, Type ty,
                               std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto i256Ty = IntegerType::get(b.getContext(), 256);
  sol::DataLocation dataLoc = sol::getDataLocation(ty);
  if (dataLoc == sol::DataLocation::CallData)
    return b.create<LLVM::ExtractValueOp>(loc, i256Ty, addr,
                                          b.getDenseI64ArrayAttr({1}));

  Value sizeSlot = genLoad(addr, dataLoc, loc);
  if (isa<sol::StringType>(ty) && dataLoc == sol::DataLocation::Storage)
    return genStorageStringLength(sizeSlot, loc);

  return sizeSlot;
}

Value evm::Builder::genDataAddrPtr(Value addr, Type ty,
                                   std::optional<Location> locArg) {
  sol::DataLocation dataLoc = sol::getDataLocation(ty);
  return genDataAddrPtr(addr, dataLoc, locArg);
}

Value evm::Builder::genDataAddrPtr(Value addr, sol::DataLocation dataLoc,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (dataLoc == sol::DataLocation::CallData) {
    auto i256Ty = IntegerType::get(b.getContext(), 256);
    assert(isa<LLVM::LLVMStructType>(addr.getType()));
    return b.create<LLVM::ExtractValueOp>(loc, i256Ty, addr,
                                          b.getDenseI64ArrayAttr({0}));
  }

  if (dataLoc == sol::DataLocation::Memory) {
    // Return the address after the first word.
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
  }

  if (dataLoc == sol::DataLocation::Storage) {
    // Return the keccak256 of addr.
    auto zero = bExt.genI256Const(0);
    b.create<yul::MStoreOp>(loc, zero, addr);
    return b.create<yul::Keccak256Op>(loc, zero, bExt.genI256Const(32));
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genAddrAtIdx(Value baseAddr, Value idx, Type ty,
                                 sol::DataLocation dataLoc,
                                 std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (dataLoc == sol::DataLocation::Memory) {
    Value memIdx = b.create<arith::MulIOp>(loc, idx, bExt.genI256Const(32));
    return b.create<arith::AddIOp>(loc, baseAddr, memIdx);
  }

  if (dataLoc == sol::DataLocation::Storage) {
    Value stride;
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
      stride = bExt.genI256Const(evm::getStorageSlotCount(arrTy.getEltType()));
    else if (isa<sol::StringType>(ty))
      stride = bExt.genI256Const(1);
    Value scaledIdx = b.create<arith::MulIOp>(loc, idx, stride);
    return b.create<arith::AddIOp>(loc, baseAddr, scaledIdx);
  }

  llvm_unreachable("NYI");
}

static std::pair<Value, Value>
genPackedStorageAddrPair(OpBuilder b, Value baseSlot, Value idx,
                         unsigned eltByteSize, bool isDataLeftAligned,
                         Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);

  Value bytePos =
      b.create<arith::MulIOp>(loc, idx, bExt.genI256Const(eltByteSize, loc));
  Value thirtyTwo = bExt.genI256Const(32);
  Value slotOffset = b.create<arith::DivUIOp>(loc, bytePos, thirtyTwo);
  Value byteOffset = b.create<arith::RemUIOp>(loc, bytePos, thirtyTwo);
  if (isDataLeftAligned)
    byteOffset =
        b.create<arith::SubIOp>(loc, bExt.genI256Const(31, loc), byteOffset);

  Value slot = b.create<arith::AddIOp>(loc, baseSlot, slotOffset);
  return {slot, byteOffset};
}

Value evm::Builder::genPackedStorageAddr(Value baseSlot, Value idx, Type eltTy,
                                         bool isDataLeftAligned,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto [slot, byteOffset] = genPackedStorageAddrPair(
      b, baseSlot, idx, getStorageByteSize(eltTy), isDataLeftAligned, loc);
  return bExt.genLLVMStruct({slot, byteOffset});
}

static Value genPunchHoleInValue(OpBuilder &b, Value value, Value shiftBits,
                                 unsigned numBits, Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);

  // holeMask = not(ones(numBits) << shiftBits)
  APInt ones = APInt::getLowBitsSet(256, numBits);
  Value shiftedOnes =
      b.create<arith::ShLIOp>(loc, bExt.genI256Const(ones), shiftBits);
  Value holeMask = b.create<arith::XOrIOp>(
      loc, shiftedOnes, bExt.genI256Const(APInt::getAllOnes(256)));

  // and(value, holeMask)
  return b.create<arith::AndIOp>(loc, value, holeMask);
}

Value evm::Builder::genPunchHole(Value slot, Value shiftBits, unsigned numBits,
                                 sol::DataLocation dataLoc,
                                 std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  Value slotVal;
  if (dataLoc == sol::DataLocation::Transient)
    slotVal = b.create<yul::TLoadOp>(loc, slot);
  else
    slotVal = b.create<yul::SLoadOp>(loc, slot);
  return genPunchHoleInValue(b, slotVal, shiftBits, numBits, loc);
}

Value evm::Builder::genLoad(Value addr, sol::DataLocation dataLoc,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  if (dataLoc == sol::DataLocation::CallData)
    return b.create<yul::CallDataLoadOp>(loc, addr);

  if (dataLoc == sol::DataLocation::Memory ||
      dataLoc == sol::DataLocation::Immutable)
    return b.create<yul::MLoadOp>(loc, addr);

  if (dataLoc == sol::DataLocation::Storage)
    return b.create<yul::SLoadOp>(loc, addr);

  if (dataLoc == sol::DataLocation::Transient)
    return b.create<yul::TLoadOp>(loc, addr);

  llvm_unreachable("NYI");
}

void evm::Builder::genStore(Value val, Value addr, sol::DataLocation dataLoc,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  if (dataLoc == sol::DataLocation::Memory ||
      dataLoc == sol::DataLocation::Immutable) {
    b.create<yul::MStoreOp>(loc, addr, val);
  } else if (dataLoc == sol::DataLocation::Storage) {
    b.create<yul::SStoreOp>(loc, addr, val);
  } else if (dataLoc == sol::DataLocation::Transient) {
    b.create<yul::TStoreOp>(loc, addr, val);
  } else {
    llvm_unreachable("NYI");
  }
}

void evm::Builder::genStringStore(std::string const &str, Value addr,
                                  std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Generate the size store.
  b.create<yul::MStoreOp>(loc, addr, bExt.genI256Const(str.length()));

  // Store the strings in 32 byte chunks of their numerical representation.
  for (size_t i = 0; i < str.length(); i += 32) {
    // Get a substring of up to 32 bytes and pad to the right with zeros.
    std::string chunk = str.substr(i, 32);
    // Left-align (pad on right with zeros to 32 bytes)
    chunk.resize(32, '\0');
    // Convert to APInt (big-endian)
    llvm::APInt chunkVal(256, 0);
    for (int j = 0; j < 32; ++j) {
      chunkVal <<= 8;
      chunkVal |= static_cast<uint8_t>(chunk[j]);
    }
    addr = b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(32));
    b.create<yul::MStoreOp>(loc, addr, bExt.genI256Const(chunkVal));
  }
}

Value evm::Builder::genStorageStringLength(Value lengthSlot,
                                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // For the storage we follow YUL's implementation to get length:
  //   length := div(data, 2)
  //   let outOfPlaceEncoding := and(data, 1)
  //   if iszero(outOfPlaceEncoding) {
  //     length := and(length, 0x7f)
  //   }
  //
  //  if eq(outOfPlaceEncoding, lt(length, 32)) {
  //    panic_error_0x22()
  //  }
  //
  Value one = bExt.genI256Const(1);
  Value length = b.create<arith::ShRUIOp>(loc, lengthSlot, one);
  Value isOutOfPlaceEnc = b.create<arith::AndIOp>(loc, lengthSlot, one);
  Value isInPlace = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, isOutOfPlaceEnc, bExt.genI256Const(0));

  Value maskedLength =
      b.create<arith::AndIOp>(loc, length, bExt.genI256Const(0x7F));
  length = b.create<arith::SelectOp>(loc, isInPlace, maskedLength, length)
               .getResult();

  Value lengthLT32 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             length, bExt.genI256Const(32));
  Value panicCond = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq,
      bExt.genIntCast(1, /*isSigned=*/false, isOutOfPlaceEnc, loc), lengthLT32);

  genPanic(mlir::evm::PanicCode::StorageEncodingError, panicCond);

  return length;
}

/// Calculates: and(val, ones(256) << (256 - (maskLen * 8)))
/// Precondition: maskLen > 0. Shifting an i256 by its own bitwidth (maskLen=0
/// → shiftVal=256) produces poison under arith.shli semantics. All call sites
/// must guard against the empty-string case before invoking this helper.
static Value getI256MSBMaskedValue(OpBuilder &b, Value val, Value maskLen,
                                   Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);
  Value nbits =
      b.create<arith::MulIOp>(loc, maskLen, bExt.genI256Const(8, loc));
  Value shiftVal =
      b.create<arith::SubIOp>(loc, bExt.genI256Const(256, loc), nbits);
  Value mask = b.create<arith::ShLIOp>(
      loc, bExt.genI256Const(APInt::getAllOnes(256), loc), shiftVal);
  return b.create<mlir::arith::AndIOp>(loc, val, mask);
}

void evm::Builder::genCopyStringToStorage(Value src, Type ty, Value dstAddr,
                                          std::optional<Location> locArg) {
  // Storage layout for `bytes` / `string` in Solidity:
  //
  // - These types use two different encodings depending on their length.
  //
  // 1) Short form (length ≤ 31 bytes):
  //    - Entire value is stored in a single storage slot.
  //    - Data is left-aligned (stored in the high-order bytes).
  //    - The lowest-order byte stores `length * 2`.
  //    - Lowest bit = 0 indicates short form.
  //
  //    slot[p] = [ data (≤31 bytes) | padding | length * 2 ]
  //
  // 2) Long form (length ≥ 32 bytes):
  //    - Storage slot `p` stores `length * 2 + 1`.
  //    - Lowest bit = 1 indicates long form.
  //    - Actual data is stored separately starting at `keccak256(p)`.
  //    - Data occupies consecutive slots, 32 bytes per slot, left-aligned.
  //
  //    slot[p]               = length * 2 + 1
  //    slot[keccak256(p)+0]  = bytes[0..31]
  //    slot[keccak256(p)+1]  = bytes[32..63]
  //    ...
  //
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value lengthSlot = nullptr;
  Value length = nullptr;
  sol::DataLocation srcDataLoc = sol::getDataLocation(ty);
  if (srcDataLoc == sol::DataLocation::CallData ||
      srcDataLoc == sol::DataLocation::Memory) {
    length = genDynSize(src, ty, loc);
  } else {
    assert(srcDataLoc == sol::DataLocation::Storage);
    lengthSlot = genLoad(src, srcDataLoc, loc);
    length = genStorageStringLength(lengthSlot, loc);
  }

  Value oldLength = genStorageStringLength(
      genLoad(dstAddr, sol::DataLocation::Storage, loc), loc);
  // Remove the old string data by zeroing storage slots that are no longer
  // part of the new value. We do this if the old string has length > 31 bytes.
  {
    Value cleanCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                              oldLength, bExt.genI256Const(31));

    auto ifClean = b.create<scf::IfOp>(loc, cleanCond);

    b.setInsertionPointToStart(&ifClean.getThenRegion().front());
    Value dstDataArea =
        genDataAddrPtr(dstAddr, sol::DataLocation::Storage, loc);
    Value deleteStart = b.create<mlir::arith::AddIOp>(
        loc, dstDataArea, bExt.genCeilDivision<32>(length));
    Value shortStringCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, length, bExt.genI256Const(32));

    deleteStart = b.create<arith::SelectOp>(loc, shortStringCond, dstDataArea,
                                            deleteStart);
    Value deleteEnd = b.create<mlir::arith::AddIOp>(
        loc, dstDataArea, bExt.genCeilDivision<32>(oldLength));
    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genCastToIdx(deleteStart),
        /*upperBound=*/bExt.genCastToIdx(deleteEnd),
        /*step=*/bExt.genIdxConst(1),
        /*iterArgs=*/ArrayRef<Value>(),
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
          Value i256IndVar = bExt.genCastToI256(indVar);
          b.create<yul::SStoreOp>(loc, i256IndVar, bExt.genI256Const(0, loc));
          b.create<scf::YieldOp>(loc);
        });

    b.setInsertionPointAfter(ifClean);
  }

  // Handle out of place case.
  Value outOfPlaceCond = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, length, bExt.genI256Const(31, loc));

  auto ifOutOfPlace = b.create<scf::IfOp>(loc, outOfPlaceCond, true);
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());

  Value dstDataArea = genDataAddrPtr(dstAddr, sol::DataLocation::Storage, loc);
  Value srcDataAddr = genDataAddrPtr(src, srcDataLoc, loc);
  Value loopEnd = b.create<mlir::arith::AndIOp>(
      loc, length, bExt.genI256Const(~APInt(256, 0x1F), loc));

  // Copy the data in 32-byte chunks first. The loop variable is a byte offset.
  // For memory/calldata sources the byte offset indexes directly into the data
  // area. For storage sources, data is addressed by slot (32 bytes each), so
  // we divide the byte offset by 32 to obtain the slot index.
  b.create<scf::ForOp>(
      loc, /*lowerBound=*/bExt.genIdxConst(0),
      /*upperBound=*/bExt.genCastToIdx(loopEnd),
      /*step=*/bExt.genIdxConst(32),
      /*iterArgs=*/ArrayRef<Value>(),
      /*builder=*/
      [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
        Value i256IndVar = bExt.genCastToI256(indVar);
        Value slotIndVar = b.create<arith::DivUIOp>(loc, i256IndVar,
                                                    bExt.genI256Const(32, loc));
        Value srcIndVar =
            srcDataLoc == sol::DataLocation::Storage ? slotIndVar : i256IndVar;
        Value src = b.create<arith::AddIOp>(loc, srcDataAddr, srcIndVar);
        Value val = genLoad(src, srcDataLoc, loc);
        Value dst = b.create<arith::AddIOp>(loc, dstDataArea, slotIndVar);
        b.create<yul::SStoreOp>(loc, dst, val);
        b.create<scf::YieldOp>(loc);
      });

  Value residualCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, loopEnd, length);

  // Copy the remaining bytes (< 32) if the string length is not divisible
  // by 32.
  auto ifResidual = b.create<scf::IfOp>(loc, residualCond);
  {
    b.setInsertionPointToStart(&ifResidual.getThenRegion().front());
    Value residualLength = b.create<mlir::arith::AndIOp>(
        loc, length, bExt.genI256Const(0x1F, loc));
    Value slotLoopEnd =
        b.create<arith::DivUIOp>(loc, loopEnd, bExt.genI256Const(32, loc));
    Value srcResidualIdx =
        srcDataLoc == sol::DataLocation::Storage ? slotLoopEnd : loopEnd;
    Value lastVal =
        genLoad(b.create<arith::AddIOp>(loc, srcDataAddr, srcResidualIdx),
                srcDataLoc, loc);
    Value maskedVal = getI256MSBMaskedValue(b, lastVal, residualLength, loc);
    Value dst = b.create<arith::AddIOp>(loc, dstDataArea, slotLoopEnd);
    b.create<yul::SStoreOp>(loc, dst, maskedVal);
  }
  b.setInsertionPointAfter(ifResidual);

  // Store the string length.
  Value doubleLength =
      b.create<arith::MulIOp>(loc, length, bExt.genI256Const(2, loc));
  b.create<yul::SStoreOp>(loc, dstAddr,
                          b.create<mlir::arith::OrIOp>(
                              loc, doubleLength, bExt.genI256Const(1, loc)));

  // Handle in place case.
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value isNotEmptyCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, length, bExt.genI256Const(0, loc));

    auto ifIsNotEmpty = b.create<scf::IfOp>(loc, isNotEmptyCond, true);
    b.setInsertionPointToStart(&ifIsNotEmpty.getThenRegion().front());

    // For storage sources the source is also in-place encoding (length < 32),
    // so its data occupies the high 31 bytes of `lengthSlot` and `length*2`
    // sits in the low byte. getI256MSBMaskedValue keeps exactly the top
    // `length` bytes and discards the rest, which correctly strips that low
    // byte. For non-storage sources we load the raw data word directly.
    // isNotEmptyCond above ensures length > 0, satisfying the maskLen > 0
    // precondition of getI256MSBMaskedValue.
    Value val =
        srcDataLoc == sol::DataLocation::Storage
            ? lengthSlot
            : genLoad(genDataAddrPtr(src, srcDataLoc, loc), srcDataLoc, loc);
    Value maskedVal = getI256MSBMaskedValue(b, val, length, loc);
    Value doubleLength =
        b.create<arith::MulIOp>(loc, length, bExt.genI256Const(2, loc));

    Value packedData = b.create<arith::OrIOp>(loc, maskedVal, doubleLength);
    b.create<yul::SStoreOp>(loc, dstAddr, packedData);

    // String is empty
    b.setInsertionPointToStart(&ifIsNotEmpty.getElseRegion().front());
    b.create<yul::SStoreOp>(loc, dstAddr, bExt.genI256Const(0, loc));
  }
  b.setInsertionPointAfter(ifOutOfPlace);
}

Value evm::Builder::genCopyStringDataToMemory(Value src, Type ty,
                                              Value dstDataAddr,
                                              std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  sol::DataLocation srcDataLoc = sol::getDataLocation(ty);
  if (srcDataLoc == sol::DataLocation::Storage) {
    Value lengthSlot = genLoad(src, srcDataLoc, loc);
    Value length = genStorageStringLength(lengthSlot, loc);
    genCopyStringDataFromStorageToMemory(src, lengthSlot, length, dstDataAddr,
                                         loc);
    return length;
  }

  Value length = genDynSize(src, ty, loc);
  Value srcDataAddr = genDataAddrPtr(src, srcDataLoc, loc);
  if (srcDataLoc == sol::DataLocation::Memory)
    b.create<yul::MCopyOp>(loc, dstDataAddr, srcDataAddr, length);
  else if (srcDataLoc == sol::DataLocation::CallData)
    b.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr, length);
  else
    llvm_unreachable("Unexpected data location");

  return length;
}

void evm::Builder::genCopyStringDataFromStorageToMemory(
    Value src, Value lengthSlot, Value length, Value dstDataAddr,
    std::optional<Location> locArg) {
  // See 'genCopyStringToStorage' regarding the storage layout for
  // `bytes` / `string` in Solidity.

  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value one = bExt.genI256Const(1, loc);
  Value zero = bExt.genI256Const(0, loc);
  Value isOutOfPlaceEnc = b.create<arith::AndIOp>(loc, lengthSlot, one);
  Value isInPlace = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                            isOutOfPlaceEnc, zero);

  auto ifInPlace = b.create<scf::IfOp>(loc, isInPlace, true);
  // In-place path: the data bytes live in the high 31 bytes of the length slot
  // itself. Strip the low byte (the length encoding) and write one word.
  b.setInsertionPointToStart(&ifInPlace.getThenRegion().front());
  {
    Value val = b.create<arith::AndIOp>(
        loc, lengthSlot, bExt.genI256Const(~APInt(256, 0xFF), loc));
    b.create<yul::MStoreOp>(loc, dstDataAddr, val);
  }
  // Out-of-place path: copy 32-byte storage slots to memory word by word.
  b.setInsertionPointToStart(&ifInPlace.getElseRegion().front());
  {
    auto srcDataAddr = genDataAddrPtr(src, sol::DataLocation::Storage, loc);
    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/bExt.genCastToIdx(length),
        /*step=*/bExt.genIdxConst(32),
        /*iterArgs=*/ArrayRef<Value>(),
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange iterArgs) {
          Value i256IndVar = bExt.genCastToI256(indVar);
          Value slotOffset = b.create<arith::DivUIOp>(
              loc, i256IndVar, bExt.genI256Const(32, loc));
          Value src = b.create<arith::AddIOp>(loc, srcDataAddr, slotOffset);
          Value val = b.create<yul::SLoadOp>(loc, src);
          Value dst = b.create<arith::AddIOp>(loc, dstDataAddr, i256IndVar);
          b.create<yul::MStoreOp>(loc, dst, val);
          b.create<scf::YieldOp>(loc);
        });
  }
  b.setInsertionPointAfter(ifInPlace);
}

Value evm::Builder::genInsertIntToSlot(Value slot, Value offset, Value intVal,
                                       unsigned numBits,
                                       std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value bitOffset =
      b.create<arith::MulIOp>(loc, offset, bExt.genI256Const(8, loc));
  Value punchedSlot = genPunchHoleInValue(b, slot, bitOffset, numBits, loc);
  Value shiftedByte = b.create<arith::ShLIOp>(loc, intVal, bitOffset);
  return b.create<arith::OrIOp>(loc, punchedSlot, shiftedByte);
}

void evm::Builder::genPushToString(Value srcAddr, Value value,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value data = genLoad(srcAddr, sol::DataLocation::Storage, loc);
  Value oldLength = genStorageStringLength(data, loc);

  Value panicCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, oldLength,
                              bExt.genI256Const(APInt(256, 1).shl(64), loc));
  genPanic(mlir::evm::PanicCode::ResourceError, panicCond);

  Value isOutOfPlace = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, oldLength, bExt.genI256Const(31, loc));

  auto ifOutOfPlace = b.create<scf::IfOp>(loc, isOutOfPlace, true);
  // Out of place path
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());
  {
    Value newLength =
        b.create<arith::AddIOp>(loc, data, bExt.genI256Const(2, loc));
    // Update the length.
    b.create<yul::SStoreOp>(loc, srcAddr, newLength);
    Value dataPtr = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
    auto [slotNum, offset] =
        genPackedStorageAddrPair(b, dataPtr, oldLength, /*eltByteSize*/ 1,
                                 /*isDataLeftAligned*/ true, loc);
    Value slotVal = genLoad(slotNum, sol::DataLocation::Storage, loc);
    Value updatedSlot =
        genInsertIntToSlot(slotVal, offset, value, /*numBits*/ 8, loc);
    b.create<yul::SStoreOp>(loc, slotNum, updatedSlot);
  }

  // In place path
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value convertToUnpacked = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, oldLength, bExt.genI256Const(31, loc));

    auto ifConvertToUnpacked =
        b.create<scf::IfOp>(loc, convertToUnpacked, true);
    b.setInsertionPointToStart(&ifConvertToUnpacked.getThenRegion().front());
    {
      // Here we have special case when array switches from short array
      // to long array. We need to copy data.
      Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);

      Value mask = bExt.genI256Const(~APInt(256, 0xFF), loc);
      Value maskedData = b.create<arith::AndIOp>(loc, data, mask);
      Value res = b.create<arith::OrIOp>(loc, value, maskedData);
      b.create<yul::SStoreOp>(loc, dataSlot, res);
      // New length is 32, encoded as (32 * 2 + 1)
      b.create<yul::SStoreOp>(loc, srcAddr, bExt.genI256Const(65, loc));
    }

    b.setInsertionPointToStart(&ifConvertToUnpacked.getElseRegion().front());
    {
      Value offset =
          b.create<arith::SubIOp>(loc, bExt.genI256Const(31, loc), oldLength);
      Value updatedSlot =
          genInsertIntToSlot(data, offset, value, /*numBits*/ 8, loc);
      Value res =
          b.create<arith::AddIOp>(loc, updatedSlot, bExt.genI256Const(2, loc));
      b.create<yul::SStoreOp>(loc, srcAddr, res);
    }
  }
  b.setInsertionPointAfter(ifOutOfPlace);
}

Value evm::Builder::genPushVoidToString(Value srcAddr,
                                        std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value data = genLoad(srcAddr, sol::DataLocation::Storage, loc);
  Value oldLength = genStorageStringLength(data, loc);

  Value panicCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, oldLength,
                              bExt.genI256Const(APInt(256, 1).shl(64), loc));
  genPanic(mlir::evm::PanicCode::ResourceError, panicCond);

  Type i256Ty = IntegerType::get(b.getContext(), 256,
                                 IntegerType::SignednessSemantics::Signless);
  Type resTy =
      LLVM::LLVMStructType::getLiteral(b.getContext(), {i256Ty, i256Ty});
  Type bytes1Ty = mlir::sol::BytesType::get(b.getContext(), /*size*/ 1);

  Value isOutOfPlace = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, oldLength, bExt.genI256Const(31, loc));

  auto ifOutOfPlace = b.create<scf::IfOp>(loc, resTy, isOutOfPlace, true);
  // Out of place path
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());
  {
    Value newLength =
        b.create<arith::AddIOp>(loc, data, bExt.genI256Const(2, loc));
    Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage);
    // Update the length.
    b.create<yul::SStoreOp>(loc, srcAddr, newLength);
    b.create<scf::YieldOp>(loc,
                           genPackedStorageAddr(dataSlot, oldLength, bytes1Ty,
                                                /*isDataLeftAligned*/ true));
  }

  // In place path
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value convertToUnpacked = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, oldLength, bExt.genI256Const(31, loc));

    auto ifConvertToUnpacked =
        b.create<scf::IfOp>(loc, resTy, convertToUnpacked, true);
    b.setInsertionPointToStart(&ifConvertToUnpacked.getThenRegion().front());
    {
      // Here we have special case when array switches from short array
      // to long array. We need to copy data.
      Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
      Value mask = bExt.genI256Const(~APInt(256, 0xFF), loc);
      Value maskedOldData = b.create<arith::AndIOp>(loc, data, mask);
      b.create<yul::SStoreOp>(loc, dataSlot, maskedOldData);
      // New length is 32, encoded as (32 * 2 + 1)
      b.create<yul::SStoreOp>(loc, srcAddr, bExt.genI256Const(65, loc));
      b.create<scf::YieldOp>(loc,
                             genPackedStorageAddr(dataSlot, oldLength, bytes1Ty,
                                                  /*isDataLeftAligned*/ true));
    }

    b.setInsertionPointToStart(&ifConvertToUnpacked.getElseRegion().front());
    {
      Value res = b.create<arith::AddIOp>(loc, data, bExt.genI256Const(2, loc));
      b.create<yul::SStoreOp>(loc, srcAddr, res);
      b.create<scf::YieldOp>(loc,
                             genPackedStorageAddr(srcAddr, oldLength, bytes1Ty,
                                                  /*isDataLeftAligned*/ true));
    }
    b.setInsertionPointAfter(ifConvertToUnpacked);
    b.create<scf::YieldOp>(loc, ifConvertToUnpacked.getResult(0));
  }
  b.setInsertionPointAfter(ifOutOfPlace);

  return ifOutOfPlace.getResult(0);
}

Value evm::Builder::genStringItemAddress(Value srcAddr, Value idx,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Type bytes1Ty = mlir::sol::BytesType::get(b.getContext(), /*size*/ 1);
  Value data = genLoad(srcAddr, sol::DataLocation::Storage, loc);
  Value length = genStorageStringLength(data, loc);

  Value castedIdx = bExt.genIntCast(256, /*isSigned*/ false, idx);
  auto panicCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                           castedIdx, length);
  genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

  Value isOutOfPlace = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, length, bExt.genI256Const(31, loc));

  Value dataSlot =
      b.create<arith::SelectOp>(
           loc, isOutOfPlace,
           genDataAddrPtr(srcAddr, sol::DataLocation::Storage), srcAddr)
          .getResult();

  return genPackedStorageAddr(dataSlot, castedIdx, bytes1Ty,
                              /*isDataLeftAligned*/ true);
}

void evm::Builder::genPopString(Value srcAddr, Value oldData, Value length,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value convertToPacked = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, length, bExt.genI256Const(32, loc));

  auto ifConvertToPacked = b.create<scf::IfOp>(loc, convertToPacked, true);
  b.setInsertionPointToStart(&ifConvertToPacked.getThenRegion().front());
  {
    // Special case: array transitions from out-of-place (length == 32) to
    // in-place (length == 31). Copy the 31 MSB remaining bytes from the data
    // slot back into srcAddr in packed encoding.
    // The new length encoding is 31*2 = 62.
    Value dataPos = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
    Value slotData = genLoad(dataPos, sol::DataLocation::Storage, loc);
    Value maskedData = b.create<arith::AndIOp>(
        loc, slotData, bExt.genI256Const(~APInt(256, 0xFF), loc));
    b.create<yul::SStoreOp>(
        loc, srcAddr,
        b.create<arith::OrIOp>(loc, maskedData, bExt.genI256Const(62, loc)));
    b.create<yul::SStoreOp>(loc, dataPos, bExt.genI256Const(0, loc));
  }

  b.setInsertionPointToStart(&ifConvertToPacked.getElseRegion().front());
  {
    Value newLen = b.create<arith::SubIOp>(loc, length, bExt.genI256Const(1));
    Value isPacked = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, length, bExt.genI256Const(32, loc));

    auto ifPacked = b.create<scf::IfOp>(loc, isPacked, true);
    b.setInsertionPointToStart(&ifPacked.getThenRegion().front());
    {
      // In-place encoding: keep the top newLen bytes of oldData, zero the
      // rest, and write the new packed length encoding (newLen * 2).
      //
      // mask = NOT(allOnes >> (newLen * 8))
      //
      // Using a right-shift makes this safe for newLen in [0, 30]: shift is
      // newLen*8 in [0, 240], never reaching the bitwidth. When newLen == 0,
      // mask = NOT(allOnes >> 0) = 0, so the result is 0 (empty-string
      // encoding), with no undefined behavior.
      Value allOnes = bExt.genI256Const(APInt::getAllOnes(256), loc);
      Value nbits =
          b.create<arith::MulIOp>(loc, newLen, bExt.genI256Const(8, loc));
      Value mask = b.create<arith::XOrIOp>(
          loc, b.create<arith::ShRUIOp>(loc, allOnes, nbits), allOnes);
      Value maskedData = b.create<arith::AndIOp>(loc, oldData, mask);
      Value dLen =
          b.create<arith::MulIOp>(loc, newLen, bExt.genI256Const(2, loc));
      b.create<yul::SStoreOp>(loc, srcAddr,
                              b.create<arith::OrIOp>(loc, maskedData, dLen));
    }

    b.setInsertionPointToStart(&ifPacked.getElseRegion().front());
    {
      Value dataPtr = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
      auto [slotNum, offset] =
          genPackedStorageAddrPair(b, dataPtr, newLen, /*eltByteSize*/ 1,
                                   /*isDataLeftAligned*/ true, loc);
      Value slot = genLoad(slotNum, sol::DataLocation::Storage, loc);
      Value updatedSlot =
          genInsertIntToSlot(slot, offset, bExt.genI256Const(0, loc),
                             /*numBits*/ 8, loc);
      b.create<yul::SStoreOp>(loc, slotNum, updatedSlot);
      Value newData =
          b.create<arith::SubIOp>(loc, oldData, bExt.genI256Const(2));
      b.create<yul::SStoreOp>(loc, srcAddr, newData);
    }
    b.setInsertionPointAfter(ifPacked);
  }
  b.setInsertionPointAfter(ifConvertToPacked);
}

void evm::Builder::genCopy(Type ty, Value srcAddr, Value dstAddr,
                           sol::DataLocation srcDataLoc,
                           sol::DataLocation dstDataLoc,
                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    Value length, dstDataAddr, srcDataAddr;
    if (arrTy.isDynSized())
      llvm_unreachable("NYI");

    length = bExt.genI256Const(arrTy.getSize());
    dstDataAddr = dstAddr;
    srcDataAddr = srcAddr;
    Type eltTy = arrTy.getEltType();

    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/bExt.genCastToIdx(length),
        /*step=*/bExt.genIdxConst(1),
        /*initArgs=*/ValueRange{},
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
          Value i256IndVar = bExt.genCastToI256(indVar);
          Value srcAddrI =
              genAddrAtIdx(srcDataAddr, i256IndVar, arrTy, srcDataLoc, loc);
          Value dstAddrI =
              genAddrAtIdx(dstDataAddr, i256IndVar, arrTy, dstDataLoc, loc);

          // Reference type elements store pointers to heap-allocated objects.
          if (sol::isNonPtrRefType(eltTy)) {
            if (srcDataLoc == sol::DataLocation::Memory)
              srcAddrI = genLoad(srcAddrI, srcDataLoc, loc);
            if (dstDataLoc == sol::DataLocation::Memory)
              dstAddrI = genLoad(dstAddrI, dstDataLoc, loc);
          }
          genCopy(arrTy.getEltType(), srcAddrI, dstAddrI, srcDataLoc,
                  dstDataLoc, loc);
          b.create<scf::YieldOp>(loc);
        });
  } else if (isa<IntegerType>(ty)) {
    genStore(genLoad(srcAddr, srcDataLoc, loc), dstAddr, dstDataLoc, loc);
  } else if (isa<sol::StringType>(ty)) {
    if (dstDataLoc == sol::DataLocation::Storage) {
      genCopyStringToStorage(srcAddr, ty, dstAddr, loc);
    } else if (dstDataLoc == sol::DataLocation::Memory) {
      Value dstDataAddr = genDataAddrPtr(dstAddr, dstDataLoc, loc);
      Value length = genCopyStringDataToMemory(srcAddr, ty, dstDataAddr, loc);
      // Write the string length.
      genStore(length, dstAddr, dstDataLoc, loc);
    }
  } else {
    llvm_unreachable("NYI");
  }
}

void evm::Builder::genABITupleSizeAssert(TypeRange tys, Value tupleSize,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  auto shortTupleCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tupleSize,
                              bExt.genI256Const(totCallDataHeadSz));
  assert(shortTupleCond->getParentOfType<ModuleOp>());
  if (sol::isRevertStringsEnabled(shortTupleCond->getParentOfType<ModuleOp>()))
    genRevertWithMsg(shortTupleCond, "ABI decoding: tuple data too short", loc);
  else
    genRevert(shortTupleCond, loc);
}

Value evm::Builder::genABITupleEncoding(
    Type ty, Value src, Value dstAddr, bool dstAddrInTail, Value tupleStart,
    Value tailAddr, std::optional<Location> locArg,
    std::optional<sol::DataLocation> srcDataLoc) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Integer type
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    src = normalizeABIScalarForEncoding(intTy, src, loc, srcDataLoc);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Enum type
  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    src = normalizeABIScalarForEncoding(enumTy, src, loc, srcDataLoc);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Address type
  if (sol::isAddressLikeType(ty)) {
    src = normalizeABIScalarForEncoding(ty, src, loc, srcDataLoc);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Bytes type
  if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
    src = normalizeABIScalarForEncoding(bytesTy, src, loc, srcDataLoc);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Array type
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    Value thirtyTwo = bExt.genI256Const(32);
    Value dstArrAddr, srcArrAddr, size;
    if (arrTy.isDynSized()) {
      // Generate the size store.
      Value i256Size = genDynSize(src, arrTy, loc);
      assert(dstAddr == tailAddr);
      b.create<yul::MStoreOp>(loc, dstAddr, i256Size);

      size = bExt.genCastToIdx(i256Size);
      dstArrAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);
      srcArrAddr = genDataAddrPtr(src, arrTy, loc);

      // Generate the tail address update.
      Value sizeInBytes = b.create<arith::MulIOp>(
          loc, i256Size,
          bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType())));
      tailAddr = b.create<arith::AddIOp>(loc, dstArrAddr, sizeInBytes);
    } else {
      size = bExt.genIdxConst(arrTy.getSize());
      dstArrAddr = dstAddr;
      srcArrAddr = src;

      if (dstAddrInTail) {
        // Generate the tail address update.
        Value i256Size = bExt.genI256Const(arrTy.getSize());
        Value sizeInBytes = b.create<arith::MulIOp>(
            loc, i256Size,
            bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType())));
        tailAddr = b.create<arith::AddIOp>(loc, dstArrAddr, sizeInBytes);
      }
    }

    // Generate a loop to copy the array.
    auto forOp = b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/size,
        /*step=*/bExt.genIdxConst(1),
        /*initArgs=*/ValueRange{dstArrAddr, srcArrAddr, tailAddr},
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
          Value iDstAddr = initArgs[0];
          Value iSrcAddr = initArgs[1];
          Value iTailAddr = initArgs[2];

          Value srcVal = genLoad(iSrcAddr, arrTy.getDataLocation(), loc);
          Value nextTailAddr;
          if (sol::hasDynamicallySizedElt(arrTy.getEltType())) {
            if (arrTy.getDataLocation() == sol::DataLocation::CallData) {
              // Multi-dimensional dynamic arrays in calldata tracks inner
              // allocations using offsets wrt to the start array. Here `srcVal`
              // (on the rhs) is that offset.
              Value innerAddr =
                  b.create<arith::AddIOp>(loc, srcArrAddr, srcVal);
              // Construct the fat pointer.
              Value innerSize = b.create<yul::CallDataLoadOp>(loc, innerAddr);
              Value innerDataAddr =
                  b.create<arith::AddIOp>(loc, innerAddr, thirtyTwo);
              mlir::solgen::BuilderExt bExt(b, loc);
              srcVal = bExt.genLLVMStruct({innerDataAddr, innerSize});
            }

            b.create<yul::MStoreOp>(
                loc, iDstAddr,
                b.create<arith::SubIOp>(loc, iTailAddr, dstArrAddr));
            assert(dstAddrInTail);
            nextTailAddr = genABITupleEncoding(
                arrTy.getEltType(), srcVal, iTailAddr, dstAddrInTail,
                tupleStart, iTailAddr, loc, arrTy.getDataLocation());
          } else {
            nextTailAddr = genABITupleEncoding(
                arrTy.getEltType(), srcVal, iDstAddr, dstAddrInTail, tupleStart,
                iTailAddr, loc, arrTy.getDataLocation());
          }

          Value dstStride =
              bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType()));
          b.create<scf::YieldOp>(
              loc, ValueRange{b.create<arith::AddIOp>(loc, iDstAddr, dstStride),
                              b.create<arith::AddIOp>(loc, iSrcAddr, thirtyTwo),
                              nextTailAddr});
        });
    return forOp.getResult(2);
  }

  // String type
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    auto tailDataAddr =
        b.create<arith::AddIOp>(loc, tailAddr, bExt.genI256Const(32));

    // Generate the data copy.
    Value size = genCopyStringDataToMemory(src, ty, tailDataAddr, loc);
    // Generate the length field copy.
    b.create<yul::MStoreOp>(loc, tailAddr, size);
    return b.create<arith::AddIOp>(loc, tailDataAddr,
                                   bExt.genRoundUpToMultiple<32>(size));
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genABITupleEncoding(TypeRange tys, ValueRange vals,
                                        Value tupleStart,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  Value headAddr = tupleStart;
  Value tailAddr = b.create<arith::AddIOp>(
      loc, tupleStart, bExt.genI256Const(totCallDataHeadSz));
  for (auto it : llvm::zip(tys, vals)) {
    Type ty = std::get<0>(it);
    Value val = std::get<1>(it);
    if (sol::hasDynamicallySizedElt(ty)) {
      b.create<yul::MStoreOp>(
          loc, headAddr, b.create<arith::SubIOp>(loc, tailAddr, tupleStart));
      tailAddr = genABITupleEncoding(ty, val, tailAddr, /*dstAddrInTail=*/true,
                                     tupleStart, tailAddr);
    } else {
      tailAddr = genABITupleEncoding(ty, val, headAddr, /*dstAddrInTail=*/false,
                                     tupleStart, tailAddr);
    }
    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }

  return tailAddr;
}

Value evm::Builder::genABITupleEncoding(std::string const &str, Value headStart,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Generate the offset store at the head address.
  Value thirtyTwo = bExt.genI256Const(32);
  b.create<yul::MStoreOp>(loc, headStart, thirtyTwo);

  // Generate the string creation at the tail address.
  auto tailAddr = b.create<arith::AddIOp>(loc, headStart, thirtyTwo);
  genStringStore(str, tailAddr, loc);
  Value stringSize = bExt.genI256Const(
      32 + mlir::solgen::getRoundUpToMultiple<32>(str.length()));

  return b.create<arith::AddIOp>(loc, tailAddr, stringSize);
}

Value evm::Builder::genABIPackedEncoding(Type ty, Value val, Value addr,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Integer type.
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    unsigned bitWidth = intTy.getWidth();
    bool isBool = (bitWidth == 1);
    assert((isBool || bitWidth % 8 == 0) &&
           "Expected bool or byte-aligned integers");

    // bool is stored as uint8 in packed encoding.
    unsigned byteSize = isBool ? 1 : bitWidth / 8;
    Value normalized = normalizeABIScalarForEncoding(intTy, val, loc);
    if (byteSize < 32)
      normalized = b.create<arith::ShLIOp>(
          loc, normalized, bExt.genI256Const(256 - byteSize * 8));

    b.create<yul::MStoreOp>(loc, addr, normalized);
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(byteSize));
  }

  // Enum type.
  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    assert(enumTy.getMax() <= 255 &&
           "Expected enums with at most 256 elements");
    Value normalized = normalizeABIScalarForEncoding(enumTy, val, loc);
    Value shifted =
        b.create<arith::ShLIOp>(loc, normalized, bExt.genI256Const(248));
    b.create<yul::MStoreOp>(loc, addr, shifted);
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(1));
  }

  // Address type.
  if (sol::isAddressLikeType(ty)) {
    Value normalized = normalizeABIScalarForEncoding(ty, val, loc);
    Value shifted =
        b.create<arith::ShLIOp>(loc, normalized, bExt.genI256Const(96));
    b.create<yul::MStoreOp>(loc, addr, shifted);
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(20));
  }

  // Bytes type.
  if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
    Value normalized = normalizeABIScalarForEncoding(bytesTy, val, loc);
    b.create<yul::MStoreOp>(loc, addr, normalized);
    return b.create<arith::AddIOp>(loc, addr,
                                   bExt.genI256Const(bytesTy.getSize()));
  }

  // String type.
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    Value dataLen = genCopyStringDataToMemory(val, ty, addr, loc);
    return b.create<arith::AddIOp>(loc, addr, dataLen);
  }

  // Array type.
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (arrTy.getDataLocation() == sol::DataLocation::Storage)
      llvm_unreachable("NYI");

    Value size;
    Value srcArrAddr;
    if (arrTy.isDynSized()) {
      Value dynSize = genDynSize(val, arrTy, loc);
      size = bExt.genCastToIdx(dynSize);
      srcArrAddr = genDataAddrPtr(val, arrTy, loc);
    } else {
      size = bExt.genIdxConst(arrTy.getSize());
      srcArrAddr = val;
    }

    auto forOp = b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/size,
        /*step=*/bExt.genIdxConst(1),
        /*initArgs=*/ValueRange{addr, srcArrAddr},
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
          Value iDstAddr = initArgs[0];
          Value iSrcAddr = initArgs[1];
          Type eltTy = arrTy.getEltType();
          sol::DataLocation dataLoc = arrTy.getDataLocation();

          Value srcVal = genLoad(iSrcAddr, dataLoc, loc);
          if (!isa<IntegerType>(eltTy) && !isa<sol::EnumType>(eltTy) &&
              !isa<sol::BytesType>(eltTy) && !sol::isAddressLikeType(eltTy))
            llvm_unreachable("Only integer, enum, address-like, and bytes "
                             "types can be packed");

          srcVal = normalizeABIScalarForEncoding(eltTy, srcVal, loc, dataLoc);
          b.create<yul::MStoreOp>(loc, iDstAddr, srcVal);

          Value stride = bExt.genI256Const(getCallDataHeadSize(eltTy));
          Value nextDstAddr = b.create<arith::AddIOp>(loc, iDstAddr, stride);
          Value nextSrcAddr = b.create<arith::AddIOp>(loc, iSrcAddr, stride);
          b.create<scf::YieldOp>(loc, ValueRange{nextDstAddr, nextSrcAddr});
        });
    return forOp.getResult(0);
  }

  llvm_unreachable("NYI: packed encoding of this type");
}

Value evm::Builder::genABIPackedEncoding(TypeRange tys, ValueRange vals,
                                         Value addr,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  Value curAddr = addr;
  for (auto [ty, val] : llvm::zip(tys, vals))
    curAddr = genABIPackedEncoding(ty, val, curAddr, loc);
  return curAddr;
}

Value evm::Builder::genABITupleDecoding(Type ty, Value addr, bool fromMem,
                                        Value tupleStart, Value tupleEnd,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // TODO: Generate assertions for checking if addresses of reference types is
  // within the calldata.

  auto genLoad = [&](Value addr) -> Value {
    if (fromMem)
      return b.create<yul::MLoadOp>(loc, addr);
    return b.create<yul::CallDataLoadOp>(loc, addr);
  };

  // Integer type
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    Value arg = genLoad(addr);
    if (intTy.getWidth() != 256) {
      assert(intTy.getWidth() < 256);
      Value castedArg = bExt.genIntCastWithBoolCleanup(
          intTy.getWidth(), intTy.isSigned(), arg, loc);

      // Generate a revert check that checks if the decoded value is within in
      // the range of the integer type.
      auto revertCond = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ne, arg,
          bExt.genIntCast(/*width=*/256, intTy.isSigned(), castedArg));
      genRevert(revertCond, loc);
      return castedArg;
    }
    return arg;
  }

  // Enum type
  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    Value arg = genLoad(addr);
    // Generate a panic check that checks if the decoded value is within in the
    // range of the enum type.
    auto panicCond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, arg,
                                bExt.genI256Const(enumTy.getMax()));
    genPanic(mlir::evm::PanicCode::EnumConversionError, panicCond, loc);
    return arg;
  }

  // Address type
  if (sol::isAddressLikeType(ty)) {
    Value arg = genLoad(addr);
    APInt mask = APInt::getLowBitsSet(256, 160);
    Value maskedArg =
        b.create<arith::AndIOp>(loc, arg, bExt.genI256Const(mask));
    auto revertCond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, arg, maskedArg);
    genRevert(revertCond, loc);
    return maskedArg;
  }

  // Array type
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (arrTy.getDataLocation() == sol::DataLocation::CallData &&
        !arrTy.isDynSized())
      return addr;

    Value dstAddr, srcAddr, size, ret;
    Value thirtyTwo = bExt.genI256Const(32);
    if (arrTy.isDynSized()) {
      Value i256Size = genLoad(addr);
      srcAddr = b.create<arith::AddIOp>(loc, addr, thirtyTwo);

      // Generate an assertion that checks the size. (We don't need to do this
      // for static arrays because we already generated the tuple size
      // assertion).
      auto scaledSize = b.create<arith::MulIOp>(
          loc, i256Size,
          bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType())));
      auto endAddr = b.create<arith::AddIOp>(loc, srcAddr, scaledSize);
      genRevertWithMsg(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               endAddr, tupleEnd),
                       "ABI decoding: invalid array size", loc);

      if (arrTy.getDataLocation() == sol::DataLocation::CallData)
        return bExt.genLLVMStruct({srcAddr, i256Size});

      dstAddr = genMemAllocForDynArray(
          i256Size, b.create<arith::MulIOp>(loc, i256Size, thirtyTwo));
      ret = dstAddr;
      // Skip the size fields in both the addresses.
      dstAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);
      size = bExt.genCastToIdx(i256Size);
    } else {
      dstAddr = genMemAlloc(bExt.genI256Const(arrTy.getSize() * 32), loc);
      ret = dstAddr;
      srcAddr = addr;
      size = bExt.genIdxConst(arrTy.getSize());
    }

    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/size,
        /*step=*/bExt.genIdxConst(1),
        /*initArgs=*/ValueRange{dstAddr, srcAddr},
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
          Value iDstAddr = initArgs[0];
          Value iSrcAddr = initArgs[1];
          if (sol::hasDynamicallySizedElt(arrTy.getEltType())) {
            // The elements are offset wrt to the start of this array (after the
            // size field if dynamic) that contain the inner element.
            Value offsetFromSrcArr =
                b.create<arith::AddIOp>(loc, srcAddr, genLoad(iSrcAddr));
            genRevertWithMsg(
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                        offsetFromSrcArr, tupleEnd),
                "ABI decoding: invalid array offset", loc);
            b.create<yul::MStoreOp>(
                loc, iDstAddr,
                genABITupleDecoding(arrTy.getEltType(), offsetFromSrcArr,
                                    fromMem, tupleStart, tupleEnd, loc));
          } else {
            auto elemVal =
                genABITupleDecoding(arrTy.getEltType(), iSrcAddr, fromMem,
                                    tupleStart, tupleEnd, loc);
            auto intTy = dyn_cast<IntegerType>(elemVal.getType());

            // If the element type is an integer smaller than 256 bits, we need
            // to extend it. In case we don't do that, store will place the
            // value in the higher bits, which is incorrect.
            if (intTy && intTy.getWidth() < 256)
              elemVal =
                  bExt.genIntCast(/*width=*/256, intTy.isSigned(), elemVal);
            b.create<yul::MStoreOp>(loc, iDstAddr, elemVal);
          }

          Value srcStride =
              bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType()));
          b.create<scf::YieldOp>(
              loc,
              ValueRange{
                  b.create<arith::AddIOp>(loc, iDstAddr, bExt.genI256Const(32)),
                  b.create<arith::AddIOp>(loc, iSrcAddr, srcStride)});
        });
    return ret;
  }

  // Bytes type
  if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
    Value arg = genLoad(addr);
    if (bytesTy.getSize() != 32) {
      assert(bytesTy.getSize() < 32);
      unsigned numBits = bytesTy.getSize() * 8;
      Value mask = b.create<arith::ShLIOp>(
          loc, bExt.genI256Const(APInt::getMaxValue(numBits)),
          bExt.genI256Const(256 - numBits));
      Value maskedArg = b.create<arith::AndIOp>(loc, arg, mask);
      auto revertCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                arg, maskedArg);
      genRevert(revertCond, loc);
    }
    return arg;
  }

  // String type
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    Value tailAddr = addr;

    Value sizeInBytes = genLoad(tailAddr);
    Value thirtyTwo = bExt.genI256Const(32);
    Value srcDataAddr = b.create<arith::AddIOp>(loc, tailAddr, thirtyTwo);
    Value endAddr = b.create<arith::AddIOp>(loc, srcDataAddr, sizeInBytes);
    genRevertWithMsg(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             endAddr, tupleEnd),
                     "ABI decoding: invalid byte array length", loc);

    if (stringTy.getDataLocation() == sol::DataLocation::CallData)
      return bExt.genLLVMStruct({srcDataAddr, sizeInBytes});

    // Copy the decoded string to a new memory allocation.
    Value dstAddr = genMemAllocForDynArray(
        sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes), loc);
    Value dstDataAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);

    // FIXME: ABIFunctions::abiDecodingFunctionByteArrayAvailableLength only
    // allocates length + 32 (where length is rounded up to a multiple of 32)
    // bytes. The "+ 32" is for the size field. But it calls
    // YulUtilFunctions::copyToMemoryFunction with the _cleanup param enabled
    // which makes the writing of the zero at the end an out-of-bounds write.
    // Even if the allocation was done correctly, do we need to write zero at
    // the end?

    if (fromMem)
      // TODO? Check m_evmVersion.hasMcopy() and legalize here or in sol.mcopy
      // lowering?
      b.create<yul::MCopyOp>(loc, dstDataAddr, srcDataAddr, sizeInBytes);
    else
      b.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr, sizeInBytes);

    return dstAddr;
  }

  llvm_unreachable("NYI");
}

void evm::Builder::genABITupleDecoding(TypeRange tys, Value tupleStart,
                                       Value tupleEnd,
                                       std::vector<Value> &results,
                                       bool fromMem,
                                       std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // TODO? {en|de}codingType() for sol dialect types.

  genABITupleSizeAssert(tys, b.create<arith::SubIOp>(loc, tupleEnd, tupleStart),
                        loc);

  auto genLoad = [&](Value addr) -> Value {
    if (fromMem)
      return b.create<yul::MLoadOp>(loc, addr);
    return b.create<yul::CallDataLoadOp>(loc, addr);
  };

  // Decode the args.
  // The type of the decoded arg should be same as that of the legalized type
  // (as per the type-converter) of the original type.
  Value headAddr = tupleStart;
  for (Type ty : tys) {
    if (sol::hasDynamicallySizedElt(ty)) {
      // TODO: Do we need the "ABI decoding: invalid tuple offset" check here?
      Value tailAddr =
          b.create<arith::AddIOp>(loc, tupleStart, genLoad(headAddr));

      // The `tailAddr` should point to at least 1 32-byte word in the tuple.
      // Generate a revert check for that.
      auto invalidTailAddrCond = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge,
          b.create<arith::AddIOp>(loc, tailAddr, bExt.genI256Const(31)),
          tupleEnd);
      genRevertWithMsg(invalidTailAddrCond,
                       "ABI decoding: invalid calldata array offset", loc);
      results.push_back(genABITupleDecoding(ty, tailAddr, fromMem, tupleStart,
                                            tupleEnd, loc));
    } else {
      results.push_back(genABITupleDecoding(ty, headAddr, fromMem, tupleStart,
                                            tupleEnd, loc));
    }
    headAddr = b.create<arith::AddIOp>(
        loc, headAddr, bExt.genI256Const(getCallDataHeadSize(ty)));
  }
}

void evm::Builder::genPanic(PanicCode code, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  b.create<yul::MStoreOp>(loc, bExt.genI256Const(0),
                          bExt.genI256Selector("Panic(uint256)"));
  b.create<yul::MStoreOp>(loc, bExt.genI256Const(4),
                          bExt.genI256Const(static_cast<int64_t>(code)));
  b.create<yul::RevertOp>(loc, bExt.genI256Const(0), bExt.genI256Const(0x24));
}

void evm::Builder::genPanic(PanicCode code, Value cond,
                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = b.create<scf::IfOp>(loc, cond, /*addThenBlock=*/true);
  b.setInsertionPointToStart(ifOp.thenBlock());
  genPanic(code, loc);
  b.setInsertionPointAfter(ifOp);
}

void evm::Builder::genForwardingRevert(std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value freePtr = genFreePtr(loc);
  Value retDataSize = b.create<yul::ReturnDataSizeOp>(loc);
  b.create<yul::ReturnDataCopyOp>(loc, /*dst=*/freePtr,
                                  /*src=*/bExt.genI256Const(0), retDataSize);
  b.create<yul::RevertOp>(loc, freePtr, retDataSize);
}

void evm::Builder::genForwardingRevert(Value cond,
                                       std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genForwardingRevert(loc);
}

void evm::Builder::genRevert(Value cond, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());

  mlir::solgen::BuilderExt bExt(b, loc);
  mlir::Value zero = bExt.genI256Const(0);
  b.create<yul::RevertOp>(loc, zero, zero);
}

void evm::Builder::genRevert(TypeRange tys, ValueRange vals,
                             StringRef signature,
                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value selectorAddr = genFreePtr(loc);
  b.create<yul::MStoreOp>(loc, selectorAddr, bExt.genI256Selector(signature));
  Value tupleStart =
      b.create<arith::AddIOp>(loc, selectorAddr, bExt.genI256Const(4));
  Value tupleEnd = genABITupleEncoding(tys, vals, tupleStart, loc);
  Value size = b.create<arith::SubIOp>(loc, tupleEnd, selectorAddr);
  b.create<yul::RevertOp>(loc, selectorAddr, size);
}

void evm::Builder::genRevert(Value cond, TypeRange tys, ValueRange vals,
                             StringRef signature,
                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevert(tys, vals, signature, loc);
}

void evm::Builder::genRevertWithMsg(std::string const &msg,
                                    std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  mlir::solgen::BuilderExt bExt(b, loc);

  // Generate the "Error(string)" selector store at free ptr.
  Value freePtr = genFreePtr(loc);
  b.create<yul::MStoreOp>(loc, freePtr, bExt.genI256Selector("Error(string)"));

  // Generate the tuple encoding of the message after the selector.
  auto freePtrPostSelector =
      b.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(4));
  Value tailAddr =
      genABITupleEncoding(msg, /*headStart=*/freePtrPostSelector, loc);

  // Generate the revert.
  auto retDataSize = b.create<arith::SubIOp>(loc, tailAddr, freePtr);
  b.create<yul::RevertOp>(loc, freePtr, retDataSize);
}

void evm::Builder::genRevertWithMsg(Value cond, std::string const &msg,
                                    std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevertWithMsg(msg, loc);
}

void evm::Builder::genDbgRevert(ValueRange vals,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value freePtr = genFreePtr(loc);
  unsigned retDataSize = 0;
  for (Value val : vals) {
    auto offset =
        b.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(retDataSize));
    b.create<yul::MStoreOp>(loc, offset, val);
    retDataSize += 32;
  }
  b.create<yul::RevertOp>(loc, freePtr, bExt.genI256Const(retDataSize));
}

void evm::Builder::genCondDbgRevert(Value cond, ValueRange vals,
                                    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  auto ifOp = b.create<scf::IfOp>(loc, cond);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genDbgRevert(vals, loc);
}

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
      isa<sol::BytesType>(ty) || sol::hasDynamicallySizedElt(ty))
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
      isa<sol::FuncRefType>(ty) || sol::hasDynamicallySizedElt(ty))
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
  return genLoad(addr, dataLoc, loc);
}

Value evm::Builder::genDataAddrPtr(Value addr, Type ty,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto i256Ty = IntegerType::get(b.getContext(), 256);
  sol::DataLocation dataLoc = sol::getDataLocation(ty);
  if (dataLoc == sol::DataLocation::CallData) {
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

Value evm::Builder::genDataAddrPtr(Value addr, sol::DataLocation dataLoc,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  assert(dataLoc != sol::DataLocation::CallData);

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

void evm::Builder::genCopyLoop(Value srcAddr, Value dstAddr, Value sizeInWords,
                               Type srcTy, Type dstTy,
                               sol::DataLocation srcDataLoc,
                               sol::DataLocation dstDataLoc,
                               std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  mlir::solgen::BuilderExt bExt(b, loc);

  b.create<scf::ForOp>(
      loc, /*lowerBound=*/bExt.genIdxConst(0),
      /*upperBound=*/bExt.genCastToIdx(sizeInWords),
      /*step=*/bExt.genIdxConst(1),
      /*initArgs=*/ValueRange{},
      /*builder=*/
      [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
        Value i256IndVar = bExt.genCastToI256(indVar);

        Value srcAddrAtIdx =
            genAddrAtIdx(srcAddr, i256IndVar, srcTy, srcDataLoc, loc);
        Value val = genLoad(srcAddrAtIdx, srcDataLoc, loc);
        Value dstAddrAtIdx =
            genAddrAtIdx(dstAddr, i256IndVar, dstTy, dstDataLoc, loc);
        genStore(val, dstAddrAtIdx, dstDataLoc, loc);

        b.create<scf::YieldOp>(loc);
      });
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

Value evm::Builder::genABITupleEncoding(Type ty, Value src, Value dstAddr,
                                        bool dstAddrInTail, Value tupleStart,
                                        Value tailAddr,
                                        std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Integer type
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    src = bExt.genIntCast(/*width=*/256, intTy.isSigned(), src);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Enum type
  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    src = bExt.genIntCast(/*width=*/256, /*isSigned=*/false, src);
    b.create<yul::MStoreOp>(loc, dstAddr, src);
    return tailAddr;
  }

  // Bytes type
  if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
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
      Value sizeInBytes = b.create<arith::MulIOp>(loc, i256Size, thirtyTwo);
      tailAddr = b.create<arith::AddIOp>(loc, dstArrAddr, sizeInBytes);
    } else {
      size = bExt.genIdxConst(arrTy.getSize());
      dstArrAddr = dstAddr;
      srcArrAddr = src;

      if (dstAddrInTail) {
        // Generate the tail address update.
        Value i256Size = bExt.genI256Const(arrTy.getSize());
        Value sizeInBytes = b.create<arith::MulIOp>(loc, i256Size, thirtyTwo);
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
            nextTailAddr =
                genABITupleEncoding(arrTy.getEltType(), srcVal, iTailAddr,
                                    dstAddrInTail, tupleStart, iTailAddr, loc);
          } else {
            nextTailAddr =
                genABITupleEncoding(arrTy.getEltType(), srcVal, iDstAddr,
                                    dstAddrInTail, tupleStart, iTailAddr, loc);
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
    // Generate the length field copy.
    auto size = genDynSize(src, stringTy, loc);
    b.create<yul::MStoreOp>(loc, tailAddr, size);

    // Generate the data copy.
    auto dataAddr = genDataAddrPtr(src, stringTy, loc);
    auto tailDataAddr =
        b.create<arith::AddIOp>(loc, tailAddr, bExt.genI256Const(32));
    if (stringTy.getDataLocation() == sol::DataLocation::Memory)
      b.create<yul::MCopyOp>(loc, tailDataAddr, dataAddr, size);
    else if (stringTy.getDataLocation() == sol::DataLocation::CallData)
      b.create<yul::CallDataCopyOp>(loc, tailDataAddr, dataAddr, size);
    else
      llvm_unreachable("NYI");

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
      Value castedArg =
          bExt.genIntCast(intTy.getWidth(), intTy.isSigned(), arg);

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
            b.create<yul::MStoreOp>(
                loc, iDstAddr,
                genABITupleDecoding(arrTy.getEltType(), iSrcAddr, fromMem,
                                    tupleStart, tupleEnd, loc));
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

    // Copy the decoded string to a new memory allocation.
    Value sizeInBytes = genLoad(tailAddr);
    Value dstAddr = genMemAllocForDynArray(
        sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes), loc);
    Value thirtyTwo = bExt.genI256Const(32);
    Value dstDataAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);
    Value srcDataAddr = b.create<arith::AddIOp>(loc, tailAddr, thirtyTwo);
    Value endAddr = b.create<arith::AddIOp>(loc, srcDataAddr, sizeInBytes);
    genRevertWithMsg(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             endAddr, tupleEnd),
                     "ABI decoding: invalid byte array length", loc);

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

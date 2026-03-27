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
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"

using namespace mlir;

namespace {

// Returns the minimum inline head that must be readable after following a
// calldata offset to a struct value. This is the sum of the member head sizes,
// which differs from getCallDataHeadSize(), as the outer head contributes one
// 32-byte offset slot for a dynamically encoded struct, but the referenced
// struct tail still begins with the full inline head of its members.
unsigned getStructCalldataEncodedTailSize(sol::StructType structTy) {
  unsigned structHeadSize = 0;
  for (Type memTy : structTy.getMemberTypes())
    structHeadSize += evm::getCallDataHeadSize(memTy);
  return structHeadSize;
}

// Returns the minimum inline head that must be readable after following a
// calldata offset to this reference type. This is computed for the referenced
// type itself, not necessarily for the top-level parameter type being encoded.
//
// Examples:
// - `uint256[]` and `string` need 32 bytes, because their tails start with a
//   length word.
// - `uint256[][2]` needs 64 bytes when that type is itself reached through an
//   offset, for example as `struct S { uint256[][2] a; }` or as an element of
//   `uint256[][2][]`. The referenced fixed-size array tail starts with two
//   32-byte element head entries.
// - `uint256[2][]` needs 32 bytes, because the referenced dynamic array tail
//   starts with its length word.
// - Structs use the sum of their member head sizes.
unsigned getCalldataEncodedTailSize(Type ty) {
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (arrTy.isDynSized())
      return 32;
    return arrTy.getSize() * evm::getCallDataHeadSize(arrTy.getEltType());
  }
  if (auto structTy = dyn_cast<sol::StructType>(ty))
    return getStructCalldataEncodedTailSize(structTy);
  if (isa<sol::StringType>(ty))
    return 32;
  llvm_unreachable("Expected dynamically encoded calldata reference type");
}

// Resolves a calldata reference reached through an offset-bearing head entry.
// Fixed-size refs with dynamic children still lower to a single base address,
// while dynamically sized refs materialize {data, length}.
Value genCalldataAccessRef(evm::Builder &evmB, OpBuilder &b, Location loc,
                           Type ty, Value baseAddr, Value ptr) {
  mlir::solgen::BuilderExt bExt(b, loc);
  assert(sol::getDataLocation(ty) == sol::DataLocation::CallData &&
         "Expected calldata reference type");
  assert(sol::hasDynamicallySizedElt(ty) &&
         "Expected dynamically encoded calldata reference");

  // Dynamically encoded calldata references are stored as offsets relative to
  // the start of the enclosing calldata head.
  Value relOffset = evmB.genLoad(ptr, sol::DataLocation::CallData, loc);
  unsigned neededLength = getCalldataEncodedTailSize(ty);
  assert(neededLength > 0 && "Expected non-zero calldata access size");

  Value callDataSize = b.create<yul::CallDataSizeOp>(loc);

  // The referenced head must fit entirely within the original calldata blob.
  // 'maxRelOffset' is the last relative offset whose required inline head is
  // still readable from 'baseAddr'.
  Value maxRelOffset = b.create<arith::SubIOp>(
      loc, b.create<arith::SubIOp>(loc, callDataSize, baseAddr),
      bExt.genI256Const(neededLength - 1));
  Value invalidOffset = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                relOffset, maxRelOffset);
  evmB.genRevertWithMsg(invalidOffset, "Invalid calldata access offset", loc);

  Value refAddr = b.create<arith::AddIOp>(loc, baseAddr, relOffset);
  auto genLengthAndStrideGuards = [&](Value dataAddr, Value length,
                                      Value stride) {
    // Dynamic calldata payloads are materialized as {dataPtr, length}, so both
    // the decoded length and the implied payload range must stay in bounds.
    Value invalidLength = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, length,
        bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    evmB.genRevertWithMsg(invalidLength, "Invalid calldata access length", loc);

    Value callDataSize = b.create<yul::CallDataSizeOp>(loc);
    Value maxDataAddr = b.create<arith::SubIOp>(
        loc, callDataSize, b.create<arith::MulIOp>(loc, length, stride));
    Value invalidStride = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, dataAddr, maxDataAddr);
    evmB.genRevertWithMsg(invalidStride, "Invalid calldata access stride", loc);
  };

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (!arrTy.isDynSized())
      // Fixed-size calldata arrays with dynamic children are still forwarded as
      // a single base address, as recursive element handling resolves child
      // heads.
      return refAddr;

    // Dynamic calldata arrays are passed as {dataPtr, size}.
    Value length = b.create<yul::CallDataLoadOp>(loc, refAddr);
    Value dataAddr =
        b.create<arith::AddIOp>(loc, refAddr, bExt.genI256Const(32));
    genLengthAndStrideGuards(
        dataAddr, length,
        bExt.genI256Const(evm::getCallDataHeadSize(arrTy.getEltType())));
    return bExt.genLLVMStruct({dataAddr, length});
  }

  if (isa<sol::StringType>(ty)) {
    // Strings use the same {dataPtr, size} representation as bytes arrays.
    Value length = b.create<yul::CallDataLoadOp>(loc, refAddr);
    Value dataAddr =
        b.create<arith::AddIOp>(loc, refAddr, bExt.genI256Const(32));
    genLengthAndStrideGuards(dataAddr, length, bExt.genI256Const(1));
    return bExt.genLLVMStruct({dataAddr, length});
  }

  if (isa<sol::StructType>(ty))
    // Nested calldata structs are forwarded by their head address.
    return refAddr;

  llvm_unreachable("Unexpected dynamically encoded calldata reference");
}

// Reads struct members for ABI encoding from a concrete source location
// (calldata, memory, or storage).
struct StructEncodeMemberReader {
  sol::StructType structTy;
  evm::Builder &evmB;
  OpBuilder &b;
  Location loc;
  mlir::solgen::BuilderExt bExt;

  StructEncodeMemberReader(sol::StructType structTy, evm::Builder &evmB,
                           OpBuilder &b, Location loc)
      : structTy(structTy), evmB(evmB), b(b), loc(loc), bExt(b, loc) {}

  virtual ~StructEncodeMemberReader() = default;

  Type getMemberType(uint64_t memberIdx) const {
    return structTy.getMemberTypes()[memberIdx];
  }

  // Returns the source value for the struct member at memberIdx.
  virtual Value read(uint64_t memberIdx) = 0;

  // Emits instruction to advance source position for the next member,
  // if needed.
  virtual void advance(uint64_t memberIdx) = 0;
};

struct StructEncodeMemberReaderCallData final : StructEncodeMemberReader {
  // Base address of this struct head in calldata.
  Value baseAddr;
  // Current calldata head cursor for member traversal.
  Value curAddr;

  StructEncodeMemberReaderCallData(sol::StructType structTy, evm::Builder &evmB,
                                   OpBuilder &b, Location loc, Value baseAddr)
      : StructEncodeMemberReader(structTy, evmB, b, loc), baseAddr(baseAddr),
        curAddr(baseAddr) {}

  Value read(uint64_t memberIdx) override {
    Type memTy = getMemberType(memberIdx);
    if (sol::hasDynamicallySizedElt(memTy))
      return genCalldataAccessRef(evmB, b, loc, memTy, baseAddr, curAddr);

    if (sol::isNonPtrRefType(memTy))
      // Static non-pointer refs are laid out inline in calldata head and
      // should be passed by address.
      return curAddr;

    return evmB.genLoad(curAddr, sol::DataLocation::CallData, loc);
  }

  void advance(uint64_t memberIdx) override {
    Type memTy = getMemberType(memberIdx);
    curAddr = b.create<arith::AddIOp>(
        loc, curAddr, bExt.genI256Const(evm::getCallDataHeadSize(memTy)));
  }
};

struct StructEncodeMemberReaderMemory final : StructEncodeMemberReader {
  // Current memory head cursor for member traversal.
  Value curAddr;

  StructEncodeMemberReaderMemory(sol::StructType structTy, evm::Builder &evmB,
                                 OpBuilder &b, Location loc, Value curAddr)
      : StructEncodeMemberReader(structTy, evmB, b, loc), curAddr(curAddr) {}

  Value read(uint64_t) override {
    Value srcVal = evmB.genLoad(curAddr, sol::DataLocation::Memory, loc);
    return srcVal;
  }

  void advance(uint64_t) override {
    // In memory, each struct member head entry occupies one 32-byte slot.
    curAddr = b.create<arith::AddIOp>(loc, curAddr, bExt.genI256Const(32));
  }
};

struct StructEncodeMemberReaderStorage final : StructEncodeMemberReader {
  // Base storage slot for the struct.
  Value baseAddr;

  // Cache the last loaded storage slot to avoid repeated sload for packed
  // members that share the same slot.
  std::optional<uint64_t> previousSlotOffset;
  Value previousSlotValue;

  StructEncodeMemberReaderStorage(sol::StructType structTy, evm::Builder &evmB,
                                  OpBuilder &b, Location loc, Value baseAddr)
      : StructEncodeMemberReader(structTy, evmB, b, loc), baseAddr(baseAddr) {}

  Value read(uint64_t memberIdx) override {
    auto [slotOffset, byteOffset] = structTy.getStorageMemberOffset(memberIdx);
    Type memTy = getMemberType(memberIdx);
    if (sol::canBePacked(memTy)) {
      // Packed members are extracted from the resolved slot and cleaned to the
      // source type width.
      if (!previousSlotOffset || *previousSlotOffset != slotOffset) {
        Value memberSlot = b.create<arith::AddIOp>(
            loc, baseAddr, bExt.genI256Const(slotOffset));
        previousSlotValue =
            evmB.genLoad(memberSlot, sol::DataLocation::Storage, loc);
        previousSlotOffset = slotOffset;
      }

      Value shifted = previousSlotValue;
      if (byteOffset != 0)
        shifted = b.create<arith::ShRUIOp>(loc, previousSlotValue,
                                           bExt.genI256Const(byteOffset * 8));

      return evmB.genCleanupPackedStorageValue(memTy, shifted, loc);
    }

    return b.create<arith::AddIOp>(loc, baseAddr,
                                   bExt.genI256Const(slotOffset));
  }

  // Storage traversal is indexed by member number and precomputed offsets.
  void advance(uint64_t) override {}
};

// Copies elements from source (storage, calldata, or memory)
// to destination memory.
// Pseudo C:
//   dst = dstStart;
//   src = srcStart;
//   for (size_t i = 0; i < size; ++i) {
//     emitElement(src, dst);
//     dst += dstStride;
//     src += srcStride;
//   }
template <typename EmitElementFuncT>
Value emitLinearArrayLoop(OpBuilder &b, Location loc, Value size,
                          Value dstStart, Value srcStart, Value dstStride,
                          Value srcStride, EmitElementFuncT &&emitElement) {
  mlir::solgen::BuilderExt bExt(b, loc);
  auto forOp = b.create<scf::ForOp>(
      loc, /*lowerBound=*/bExt.genIdxConst(0),
      /*upperBound=*/size,
      /*step=*/bExt.genIdxConst(1),
      /*initArgs=*/ValueRange{dstStart, srcStart},
      /*builder=*/
      [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
        Value iDstAddr = initArgs[0];
        Value iSrcAddr = initArgs[1];

        emitElement(b, loc, iSrcAddr, iDstAddr);

        Value nextDstAddr = b.create<arith::AddIOp>(loc, iDstAddr, dstStride);
        Value nextSrcAddr = b.create<arith::AddIOp>(loc, iSrcAddr, srcStride);
        b.create<scf::YieldOp>(loc, ValueRange{nextDstAddr, nextSrcAddr});
      });
  return forOp.getResult(0);
}

// Copies packed elements from source storage slots to destination memory.
// Pseudo C:
//   const size_t itemsPerSlot = 32 / eltByteSize;
//   const size_t fullLoopUpperBound =
//       (size >= itemsPerSlot) ? (size - (itemsPerSlot - 1)) : 0;
//   dst = dstStart;
//   srcSlot = srcStart;
//   itemCounter = 0;
//
//   // Full slot loop.
//   for (; itemCounter < fullLoopUpperBound; itemCounter += itemsPerSlot) {
//     slot = sload(srcSlot);
//     for (size_t packedIdx = 0, shiftBits = 0; packedIdx < itemsPerSlot;
//          ++packedIdx, shiftBits += eltByteSize * 8) {
//       emitElement(slot, shiftBits, dst);
//       dst += dstStride;
//     }
//     ++srcSlot;
//   }
//
//   if (itemCounter < size) {
//     slot = sload(srcSlot);
//
//     // Remainder loop.
//     for (size_t remIdx = itemCounter, shiftBits = 0; remIdx < size;
//          ++remIdx, shiftBits += eltByteSize * 8) {
//       emitElement(slot, shiftBits, dst);
//       dst += dstStride;
//     }
//   }
template <typename EmitElementFromSlotFuncT>
Value emitCompactStorageArrayLoop(OpBuilder &b, Location loc, Value size,
                                  Value dstStart, Value srcStart,
                                  Value dstStride, unsigned eltByteSize,
                                  bool isDynSized,
                                  EmitElementFromSlotFuncT &&emitElement) {
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned itemsPerSlot = 32 / eltByteSize;
  assert(itemsPerSlot > 1 && "Expected at least two packed items per slot");

  // Reused constants.
  Value zeroIdx = bExt.genIdxConst(0);
  Value oneIdx = bExt.genIdxConst(1);
  Value itemsPerSlotIdx = bExt.genIdxConst(itemsPerSlot);
  Value zeroI256 = bExt.genI256Const(0);

  // TODO: If we see some perf regressions, we can change this computation, as
  // LLVM will produce llvm.usub.sat.i256 for dynamic arrays which is
  // not efficient.
  auto hasFullSlots = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                              size, itemsPerSlotIdx);
  Value fullLoopUpperBound = b.create<arith::SelectOp>(
      loc, hasFullSlots,
      b.create<arith::SubIOp>(loc, size, bExt.genIdxConst(itemsPerSlot - 1)),
      zeroIdx);

  // Add loop annotation to fully unroll.
  auto fullUnroll = LLVM::LoopUnrollAttr::get(
      b.getContext(), /*disable=*/{}, /*count=*/{},
      /*runtimeDisable=*/{}, /*full=*/b.getBoolAttr(true),
      /*followupUnrolled=*/{}, /*followupRemainder=*/{},
      /*followupAll=*/{});
  auto fullUnrollLoopAnnotation = LLVM::LoopAnnotationAttr::get(
      b.getContext(), /*disableNonforced=*/{}, /*vectorize=*/{},
      /*interleave=*/{}, /*unroll=*/fullUnroll,
      /*unrollAndJam=*/{}, /*licm=*/{}, /*distribute=*/{},
      /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
      /*mustProgress=*/{}, /*isVectorized=*/{}, /*startLoc=*/{},
      /*endLoc=*/{}, /*parallelAccesses=*/{});

  // Full-slot loop.
  auto fullSlotForOp = b.create<scf::ForOp>(
      loc, /*lowerBound=*/zeroIdx, /*upperBound=*/fullLoopUpperBound,
      /*step=*/itemsPerSlotIdx,
      /*initArgs=*/ValueRange{dstStart, srcStart, zeroIdx},
      [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
        Value iDstAddr = initArgs[0];
        Value iSrcSlot = initArgs[1];
        Value iItemCounter = initArgs[2];

        // Load the slot once and extract all packed elements from it.
        Value slotVal = b.create<yul::SLoadOp>(loc, iSrcSlot);
        auto perSlotForOp = b.create<scf::ForOp>(
            loc, /*lowerBound=*/zeroIdx, /*upperBound=*/itemsPerSlotIdx,
            /*step=*/oneIdx, /*initArgs=*/ValueRange{iDstAddr, zeroI256},
            [&](OpBuilder &b, Location loc, Value, ValueRange perSlotArgs) {
              Value jDstAddr = perSlotArgs[0];
              Value jShiftBits = perSlotArgs[1];

              emitElement(b, loc, slotVal, jShiftBits, jDstAddr);

              // Advance destination and extraction offset for next item
              // in this same slot.
              Value nextDstAddr =
                  b.create<arith::AddIOp>(loc, jDstAddr, dstStride);
              Value nextShiftBits = b.create<arith::AddIOp>(
                  loc, jShiftBits, bExt.genI256Const(eltByteSize * 8));
              b.create<scf::YieldOp>(loc,
                                     ValueRange{nextDstAddr, nextShiftBits});
            });

        // TODO: We can enable this only for -O3, and disable for rest
        // of the optimization levels to reduce code size.
        perSlotForOp->setAttr("loop_annotation", fullUnrollLoopAnnotation);

        // After consuming one full slot, move to the next storage slot
        // and advance the emitted item counter by itemsPerSlot.
        Value nextSrcSlot =
            b.create<arith::AddIOp>(loc, iSrcSlot, bExt.genI256Const(1));
        Value nextItemCounter =
            b.create<arith::AddIOp>(loc, iItemCounter, itemsPerSlotIdx);
        b.create<scf::YieldOp>(loc, ValueRange{perSlotForOp.getResult(0),
                                               nextSrcSlot, nextItemCounter});
      });

  Value dstAfterFullSlots = fullSlotForOp.getResult(0);
  Value srcSlotAfterFullSlots = fullSlotForOp.getResult(1);
  Value itemCounterAfterFullSlots = fullSlotForOp.getResult(2);

  // Remainder phase handles tail elements in a partially-filled slot, if any.
  // It reuses one final slot load and emits only the still-missing items.
  auto hasRem = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                        itemCounterAfterFullSlots, size);
  auto remIf =
      b.create<scf::IfOp>(loc, TypeRange{dstAfterFullSlots.getType()}, hasRem,
                          /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&remIf.getThenRegion().front());

    // Load exactly the slot that contains the remaining tail elements.
    Value slotVal = b.create<yul::SLoadOp>(loc, srcSlotAfterFullSlots);

    // Remainder loop.
    // TODO: For dynamic arrays, solc unrolls this loop manually with
    // conditions. If we see some perf regressions, we can do the same.
    auto remForOp = b.create<scf::ForOp>(
        loc, /*lowerBound=*/itemCounterAfterFullSlots,
        /*upperBound=*/size, /*step=*/oneIdx,
        /*initArgs=*/ValueRange{dstAfterFullSlots, zeroI256},
        [&](OpBuilder &b, Location loc, Value, ValueRange remArgs) {
          Value remDstAddr = remArgs[0];
          Value remShiftBits = remArgs[1];

          emitElement(b, loc, slotVal, remShiftBits, remDstAddr);

          // Advance to next output element and next packed field in the
          // same loaded remainder slot.
          Value nextDstAddr =
              b.create<arith::AddIOp>(loc, remDstAddr, dstStride);
          Value nextShiftBits = b.create<arith::AddIOp>(
              loc, remShiftBits, bExt.genI256Const(eltByteSize * 8));
          b.create<scf::YieldOp>(loc, ValueRange{nextDstAddr, nextShiftBits});
        });

    // TODO: We can enable this only for -O3, and disable for rest
    // of the optimization levels to reduce code size.
    if (!isDynSized)
      remForOp->setAttr("loop_annotation", fullUnrollLoopAnnotation);

    b.create<scf::YieldOp>(loc, remForOp.getResult(0));
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&remIf.getElseRegion().front());
    b.create<scf::YieldOp>(loc, dstAfterFullSlots);
  }

  return remIf.getResult(0);
}

// Returns true when a calldata array element type can be copied directly as a
// 32-byte word in the fast-path array copy.
bool canFastCopyCalldataArray(Type eltTy) {
  if (auto intTy = dyn_cast<IntegerType>(eltTy))
    // TODO: Can we allow signed integers here as well?
    return intTy.getWidth() == 256 && !intTy.isSigned();

  auto bytesTy = dyn_cast<sol::BytesType>(eltTy);
  return bytesTy && bytesTy.getSize() == 32;
}
} // namespace

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
      isa<sol::BytesType>(ty) || isa<sol::ExtFuncRefType>(ty) ||
      sol::hasDynamicallySizedElt(ty) || sol::isAddressLikeType(ty))
    return 32;

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
    return arrTy.getSize() * getCallDataHeadSize(arrTy.getEltType());

  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    unsigned size = 0;
    for (Type memTy : structTy.getMemberTypes())
      size += getCallDataHeadSize(memTy);
    return size;
  }

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

  // ExtFuncRef: MSB-aligned like bytes24. Low 64 bits must be zero.
  if (isa<sol::ExtFuncRefType>(ty)) {
    APInt mask = APInt::getHighBitsSet(256, 192);
    Value normalized =
        b.create<arith::AndIOp>(loc, val, bExt.genI256Const(mask));
    if (fromCalldata) {
      Value revertCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 val, normalized);
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

  Value newFreePtr = b.create<arith::AddIOp>(
      loc, freePtr, bExt.genRoundUpToMultiple<32>(size));

  Value isTooLarge =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, newFreePtr,
                              bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
  Value overflowed = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                             newFreePtr, freePtr);
  Value panicCond = b.create<arith::OrIOp>(loc, isTooLarge, overflowed);
  genPanic(mlir::evm::PanicCode::ResourceError, panicCond, loc);

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
                                           std::optional<Location> locArg,
                                           bool genLengthPanicGuard) {
  Location loc = locArg ? *locArg : defLoc;

  mlir::solgen::BuilderExt bExt(b, loc);

  if (genLengthPanicGuard) {
    auto panicCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, sizeVar,
        bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    genPanic(mlir::evm::PanicCode::ResourceError, panicCond, loc);
  }

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
        // Zero-extend values before storing them, as skipping this step may
        // lead to incorrect results. For example, uint8 is stored via mstore8
        // instead of mstore.
        Value val256 = bExt.genIntCast(256, /*isSigned=*/false, val);
        b.create<yul::MStoreOp>(loc, addr, val256);
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

  if (dataLoc == sol::DataLocation::CallData) {
    unsigned stride = 32;
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
      stride = getCallDataHeadSize(arrTy.getEltType());
    Value memIdx = b.create<arith::MulIOp>(loc, idx, bExt.genI256Const(stride));
    return b.create<arith::AddIOp>(loc, baseAddr, memIdx);
  }

  if (dataLoc == sol::DataLocation::Storage) {
    Value stride;
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
      stride = bExt.genI256Const(sol::getStorageSlotCount(arrTy.getEltType()));
    else if (isa<sol::StringType>(ty))
      stride = bExt.genI256Const(1);
    Value scaledIdx = b.create<arith::MulIOp>(loc, idx, stride);
    return b.create<arith::AddIOp>(loc, baseAddr, scaledIdx);
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genCalldataDynEltFatPtr(Value headSlotAddr,
                                            Value outerDataBase,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  Value relOffset = b.create<yul::CallDataLoadOp>(loc, headSlotAddr);
  Value innerBase = b.create<arith::AddIOp>(loc, outerDataBase, relOffset);
  Value innerLen = b.create<yul::CallDataLoadOp>(loc, innerBase);
  Value innerData =
      b.create<arith::AddIOp>(loc, innerBase, bExt.genI256Const(32));
  return bExt.genLLVMStruct({innerData, innerLen});
}

Value evm::Builder::genCalldataEltAddr(Value headSlotAddr, Value dataBase,
                                       mlir::Type eltTy,
                                       std::optional<Location> locArg) {
  assert(sol::hasDynamicallySizedElt(eltTy));
  Location loc = locArg ? *locArg : defLoc;
  if (sol::isDynamicallySized(eltTy))
    return genCalldataDynEltFatPtr(headSlotAddr, dataBase, loc);
  Value relOffset = genLoad(headSlotAddr, sol::DataLocation::CallData, loc);
  return b.create<arith::AddIOp>(loc, dataBase, relOffset);
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
      b, baseSlot, idx, sol::getStorageByteSize(eltTy), isDataLeftAligned, loc);
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

Value evm::Builder::genCleanupPackedStorageValue(
    Type eltTy, Value value, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (auto bytesTy = dyn_cast<sol::BytesType>(eltTy)) {
    unsigned numBits = bytesTy.getSize() * 8;
    if (numBits == 256)
      return value;
    return b.create<arith::ShLIOp>(loc, value,
                                   bExt.genI256Const(256 - numBits, loc));
  }

  if (auto intTy = dyn_cast<IntegerType>(eltTy))
    return bExt.genIntCastWithBoolCleanup(intTy.getWidth(), intTy.isSigned(),
                                          value, loc,
                                          /*maskBoolAsStorageByte=*/true);

  if (sol::isAddressLikeType(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 160);
    return b.create<arith::AndIOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::EnumType>(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 8);
    return b.create<arith::AndIOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::FuncRefType>(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 64);
    return b.create<arith::AndIOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::ExtFuncRefType>(eltTy))
    return b.create<arith::ShLIOp>(loc, value, bExt.genI256Const(64, loc));

  llvm_unreachable("Unexpected type for cleanup of packed storage value");
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

void evm::Builder::genClearStringStorageTail(Value dstAddr, Value oldLength,
                                             Value newLength,
                                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Only the out-of-place (long) form uses extra data slots; short strings
  // (≤31 bytes) are fully self-contained in the length slot.
  Value cleanCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                            oldLength, bExt.genI256Const(31));
  auto ifClean = b.create<scf::IfOp>(loc, cleanCond);
  b.setInsertionPointToStart(&ifClean.getThenRegion().front());
  {
    Value dstDataArea =
        genDataAddrPtr(dstAddr, sol::DataLocation::Storage, loc);

    // If the new string fits in-place (newLength < 32), all old data slots
    // can be cleared. Otherwise, preserve the first ceil(newLength/32) slots.
    Value deleteStart = b.create<arith::AddIOp>(
        loc, dstDataArea, bExt.genCeilDivision<32>(newLength));
    Value isInPlace = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                              newLength, bExt.genI256Const(32));
    deleteStart =
        b.create<arith::SelectOp>(loc, isInPlace, dstDataArea, deleteStart);

    Value deleteEnd = b.create<arith::AddIOp>(
        loc, dstDataArea, bExt.genCeilDivision<32>(oldLength));

    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genCastToIdx(deleteStart),
        /*upperBound=*/bExt.genCastToIdx(deleteEnd),
        /*step=*/bExt.genIdxConst(1), /*iterArgs=*/ArrayRef<Value>(),
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange) {
          b.create<yul::SStoreOp>(loc, bExt.genCastToI256(indVar),
                                  bExt.genI256Const(0, loc));
          b.create<scf::YieldOp>(loc);
        });
  }
  b.setInsertionPointAfter(ifClean);
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
  // Zero any out-of-place data slots that are no longer needed.
  genClearStringStorageTail(dstAddr, oldLength, length, loc);

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

Value evm::Builder::genStorageArraySlotCount(Value len, Type eltTy,
                                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (sol::canBePacked(eltTy)) {
    // Multiple elements share a slot: ceil(len / elemsPerSlot).
    unsigned byteSize = sol::getStorageByteSize(eltTy);
    unsigned elemsPerSlot = 32 / byteSize;
    Value padded =
        b.create<arith::AddIOp>(loc, len, bExt.genI256Const(elemsPerSlot - 1));
    return b.create<arith::DivUIOp>(loc, padded,
                                    bExt.genI256Const(elemsPerSlot));
  }
  // Non-packable: each element occupies getStorageSlotCount(eltTy) slots.
  unsigned slotsPerElt = sol::getStorageSlotCount(eltTy);
  if (slotsPerElt == 1)
    return len;
  return b.create<arith::MulIOp>(loc, len, bExt.genI256Const(slotsPerElt));
}

void evm::Builder::genClearStorageArrayTail(Value arraySlot,
                                            sol::ArrayType arrTy,
                                            Value startIdx, Value endIdx,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Type eltTy = arrTy.getEltType();
  // For dynamic arrays the data lives at keccak256(arraySlot); for static
  // arrays it lives directly at arraySlot.
  Value dataStart =
      arrTy.isDynSized()
          ? genDataAddrPtr(arraySlot, sol::DataLocation::Storage, loc)
          : arraySlot;

  Value needsClear =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, startIdx, endIdx);
  auto ifClear = b.create<scf::IfOp>(loc, needsClear, /*withElseRegion=*/false);
  b.setInsertionPointToStart(&ifClear.getThenRegion().front());
  {
    Value deleteStart = b.create<arith::AddIOp>(
        loc, dataStart, genStorageArraySlotCount(startIdx, eltTy, loc));
    Value deleteEnd = b.create<arith::AddIOp>(
        loc, dataStart, genStorageArraySlotCount(endIdx, eltTy, loc));
    Value numSlots = b.create<arith::SubIOp>(loc, deleteEnd, deleteStart);

    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/bExt.genCastToIdx(numSlots),
        /*step=*/bExt.genIdxConst(1), /*iterArgs=*/ArrayRef<Value>(),
        [&](OpBuilder &b, Location loc, Value i, ValueRange) {
          Value slot =
              b.create<arith::AddIOp>(loc, deleteStart, bExt.genCastToI256(i));
          if (auto innerArrTy = dyn_cast<sol::ArrayType>(eltTy);
              innerArrTy && innerArrTy.isDynSized()) {
            // Dynamic sub-array: read old length, zero the length slot, then
            // recursively clear the data area without a panic check (newIdx=0).
            Value oldLen = genLoad(slot, sol::DataLocation::Storage, loc);
            genStore(bExt.genI256Const(0, loc), slot,
                     sol::DataLocation::Storage, loc);
            genClearStorageArrayTail(slot, innerArrTy,
                                     bExt.genI256Const(0, loc), oldLen, loc);
          } else if (isa<sol::StringType>(eltTy)) {
            // string/bytes: clear out-of-place data slots first, then zero
            // the length slot itself.
            Value zero = bExt.genI256Const(0, loc);
            Value strOldLen = genStorageStringLength(
                genLoad(slot, sol::DataLocation::Storage, loc), loc);
            genClearStringStorageTail(slot, strOldLen, zero, loc);
            b.create<yul::SStoreOp>(loc, slot, zero);
          } else {
            // Scalar, packed, or fixed-size element: zero the slot directly.
            b.create<yul::SStoreOp>(loc, slot, bExt.genI256Const(0, loc));
          }
          b.create<scf::YieldOp>(loc);
        });
  }
  b.setInsertionPointAfter(ifClear);
}

void evm::Builder::genResizeDynStorageArray(Value arraySlot, Value newLen,
                                            Type eltTy,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  // Panic if newLen > type(uint64).max (ResourceError / too-large array).
  mlir::solgen::BuilderExt bExt(b, loc);
  Value panicCond = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, newLen,
      bExt.genI256Const(APInt::getLowBitsSet(256, 64), loc));
  genPanic(PanicCode::ResourceError, panicCond, loc);

  // Read the old length before overwriting it.
  Value oldLen = genLoad(arraySlot, sol::DataLocation::Storage, loc);

  // Write the new length.
  genStore(newLen, arraySlot, sol::DataLocation::Storage, loc);

  // Zero out storage slots that fall outside the new range.
  auto dynArrTy = sol::ArrayType::get(b.getContext(), /*size=*/-1, eltTy,
                                      sol::DataLocation::Storage);
  genClearStorageArrayTail(arraySlot, dynArrTy, newLen, oldLen, loc);
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

void evm::Builder::genCopy(Type srcTy, Type dstTy, Value srcAddr, Value dstAddr,
                           sol::DataLocation srcDataLoc,
                           sol::DataLocation dstDataLoc,
                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // If the source and destination are the same, skip the copy.
  if (srcAddr == dstAddr)
    return;

  if (auto dstArrTy = dyn_cast<sol::ArrayType>(dstTy)) {
    auto srcArrTy = dyn_cast<sol::ArrayType>(srcTy);
    assert(srcArrTy);
    Value length, dstDataAddr, srcDataAddr;
    if (dstArrTy.isDynSized()) {
      if (srcArrTy.isDynSized()) {
        length = genDynSize(srcAddr, srcTy);
        srcDataAddr = genDataAddrPtr(srcAddr, srcDataLoc, loc);
      } else {
        length = bExt.genI256Const(srcArrTy.getSize());
        srcDataAddr = srcAddr;
      }
      // Update dst array length.
      if (dstDataLoc == sol::DataLocation::Storage)
        genResizeDynStorageArray(dstAddr, length, dstArrTy.getEltType(), loc);
      else
        genStore(length, dstAddr, dstDataLoc, loc);

      dstDataAddr = genDataAddrPtr(dstAddr, dstDataLoc, loc);
    } else {
      // Static destination array: loop over the source length, then zero any
      // tail slots in the destination that the source doesn't cover.
      assert(srcArrTy);
      length = bExt.genI256Const(srcArrTy.getSize());
      dstDataAddr = dstAddr;
      srcDataAddr = srcAddr;
      if (dstDataLoc == sol::DataLocation::Storage &&
          srcArrTy.getSize() < dstArrTy.getSize()) {
        genClearStorageArrayTail(dstAddr, dstArrTy,
                                 bExt.genI256Const(srcArrTy.getSize()),
                                 bExt.genI256Const(dstArrTy.getSize()), loc);
      }
    }
    Type eltTy = dstArrTy.getEltType();

    b.create<scf::ForOp>(
        loc, /*lowerBound=*/bExt.genIdxConst(0),
        /*upperBound=*/bExt.genCastToIdx(length),
        /*step=*/bExt.genIdxConst(1),
        /*initArgs=*/ValueRange{},
        /*builder=*/
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
          Value i256IndVar = bExt.genCastToI256(indVar);
          Value srcAddrI =
              genAddrAtIdx(srcDataAddr, i256IndVar, srcArrTy, srcDataLoc, loc);
          Value dstAddrI =
              genAddrAtIdx(dstDataAddr, i256IndVar, dstArrTy, dstDataLoc, loc);

          // Reference type elements store pointers / offsets that need
          // resolving before recursing.
          if (sol::isNonPtrRefType(eltTy)) {
            if (srcDataLoc == sol::DataLocation::CallData &&
                sol::hasDynamicallySizedElt(eltTy))
              srcAddrI = genCalldataEltAddr(srcAddrI, srcDataAddr, eltTy, loc);
            if (srcDataLoc == sol::DataLocation::Memory)
              srcAddrI = genLoad(srcAddrI, srcDataLoc, loc);
            if (dstDataLoc == sol::DataLocation::Memory)
              dstAddrI = genLoad(dstAddrI, dstDataLoc, loc);
          }
          genCopy(srcArrTy.getEltType(), dstArrTy.getEltType(), srcAddrI,
                  dstAddrI, srcDataLoc, dstDataLoc, loc);
          b.create<scf::YieldOp>(loc);
        });
  } else if (isa<IntegerType>(dstTy)) {
    genStore(genLoad(srcAddr, srcDataLoc, loc), dstAddr, dstDataLoc, loc);
  } else if (isa<sol::StringType>(dstTy)) {
    if (dstDataLoc == sol::DataLocation::Storage) {
      genCopyStringToStorage(srcAddr, srcTy, dstAddr, loc);
    } else if (dstDataLoc == sol::DataLocation::Memory) {
      Value dstDataAddr = genDataAddrPtr(dstAddr, dstDataLoc, loc);
      Value length =
          genCopyStringDataToMemory(srcAddr, srcTy, dstDataAddr, loc);
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
    std::optional<sol::DataLocation> srcDataLoc, bool includeLengthPrefix) {
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

  // External function ref
  if (isa<sol::ExtFuncRefType>(ty)) {
    src = normalizeABIScalarForEncoding(ty, src, loc, srcDataLoc);
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
      if (includeLengthPrefix)
        b.create<yul::MStoreOp>(loc, dstAddr, i256Size);

      size = bExt.genCastToIdx(i256Size);
      dstArrAddr = includeLengthPrefix
                       ? b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo)
                       : dstAddr;
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

    Type eltTy = arrTy.getEltType();
    Value dstStride = bExt.genI256Const(getCallDataHeadSize(eltTy));
    auto dataLoc = arrTy.getDataLocation();
    auto emitArrayElementEncodingLoop = [&](Value srcStride,
                                            auto &&materializeSrcVal) {
      auto forOp = b.create<scf::ForOp>(
          loc, /*lowerBound=*/bExt.genIdxConst(0),
          /*upperBound=*/size,
          /*step=*/bExt.genIdxConst(1),
          /*initArgs=*/ValueRange{dstArrAddr, srcArrAddr, tailAddr},
          /*builder=*/
          [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
            Value iDstAddr = initArgs[0];
            Value iSrcAddr = initArgs[1];
            Value iTailAddr = initArgs[2];

            Value srcVal = materializeSrcVal(loc, iSrcAddr);
            Value nextTailAddr;
            if (sol::hasDynamicallySizedElt(eltTy)) {
              b.create<yul::MStoreOp>(
                  loc, iDstAddr,
                  b.create<arith::SubIOp>(loc, iTailAddr, dstArrAddr));
              assert(dstAddrInTail);
              nextTailAddr =
                  genABITupleEncoding(eltTy, srcVal, iTailAddr, dstAddrInTail,
                                      tupleStart, iTailAddr, loc, dataLoc);
            } else {
              nextTailAddr =
                  genABITupleEncoding(eltTy, srcVal, iDstAddr, dstAddrInTail,
                                      tupleStart, iTailAddr, loc, dataLoc);
            }

            b.create<scf::YieldOp>(
                loc,
                ValueRange{b.create<arith::AddIOp>(loc, iDstAddr, dstStride),
                           b.create<arith::AddIOp>(loc, iSrcAddr, srcStride),
                           nextTailAddr});
          });
      return forOp.getResult(2);
    };

    // Memory arrays are traversed linearly, one loaded word per element.
    if (dataLoc == sol::DataLocation::Memory)
      return emitArrayElementEncodingLoop(
          thirtyTwo, [&](Location loc, Value iSrcAddr) {
            return genLoad(iSrcAddr, dataLoc, loc);
          });

    // Calldata array traversal is layout-dependent.
    if (dataLoc == sol::DataLocation::CallData) {
      // Copy contiguous calldata payload in one operation.
      if (canFastCopyCalldataArray(eltTy)) {
        Value sizeInBytes = b.create<arith::MulIOp>(
            loc, bExt.genCastToI256(size),
            bExt.genI256Const(getCallDataHeadSize(eltTy)));
        b.create<yul::CallDataCopyOp>(loc, dstArrAddr, srcArrAddr, sizeInBytes);
        return tailAddr;
      }

      // Dynamic elements keep offset-based handling (load offset from head).
      if (sol::hasDynamicallySizedElt(eltTy))
        return emitArrayElementEncodingLoop(
            thirtyTwo, [&](Location loc, Value iSrcAddr) {
              // Each fixed-array head slot stores a relative calldata offset,
              // so resolve it against the array head before recursing.
              return genCalldataAccessRef(*this, b, loc, eltTy, srcArrAddr,
                                          iSrcAddr);
            });

      // Static aggregate/non-pointer-ref elements in calldata are forwarded by
      // element head address and advanced by calldata head size.
      if (sol::isNonPtrRefType(eltTy))
        return emitArrayElementEncodingLoop(
            bExt.genI256Const(getCallDataHeadSize(eltTy)),
            [&](Location, Value iSrcAddr) { return iSrcAddr; });

      // Scalar-like calldata elements are loaded one word per step with
      // +32-byte source progression.
      return emitArrayElementEncodingLoop(
          thirtyTwo, [&](Location loc, Value iSrcAddr) {
            return genLoad(iSrcAddr, dataLoc, loc);
          });
    }

    // All remaining array sources here are storage-backed.
    assert(dataLoc == sol::DataLocation::Storage &&
           "Expected storage data location");

    // Storage arrays with reference-typed elements pass per-element storage
    // slot addresses to recursive encoders.
    if (sol::isNonPtrRefType(eltTy))
      return emitArrayElementEncodingLoop(
          bExt.genI256Const(sol::getStorageSlotCount(eltTy)),
          [&](Location, Value iSrcAddr) { return iSrcAddr; });

    if (sol::canBePacked(eltTy) && sol::getStorageByteSize(eltTy) <= 16)
      // Storage arrays with value-typed elements.
      emitCompactStorageArrayLoop(
          b, loc, size, dstArrAddr, srcArrAddr, dstStride,
          sol::getStorageByteSize(eltTy), arrTy.isDynSized(),
          [&](OpBuilder &builder, Location loc, Value slotValue,
              Value shiftBits, Value iDstAddr) {
            Value shifted =
                builder.create<arith::ShRUIOp>(loc, slotValue, shiftBits);
            Value srcVal = genCleanupPackedStorageValue(eltTy, shifted, loc);
            (void)genABITupleEncoding(eltTy, srcVal, iDstAddr, dstAddrInTail,
                                      tupleStart, tailAddr, loc, dataLoc);
          });
    else
      // Non-packed storage value elements are loaded linearly using slot
      // stride.
      emitLinearArrayLoop(
          b, loc, size, dstArrAddr, srcArrAddr, dstStride,
          bExt.genI256Const(sol::getStorageSlotCount(eltTy)),
          [&](OpBuilder &, Location loc, Value iSrcAddr, Value iDstAddr) {
            Value srcVal = genLoad(iSrcAddr, dataLoc, loc);
            (void)genABITupleEncoding(eltTy, srcVal, iDstAddr, dstAddrInTail,
                                      tupleStart, tailAddr, loc, dataLoc);
          });
    return tailAddr;
  }

  // Struct type
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    // If the struct itself is emitted into tail, initialize the struct-local
    // tail to just past its ABI head.
    if (dstAddrInTail)
      tailAddr = b.create<arith::AddIOp>(
          loc, dstAddr,
          bExt.genI256Const(getStructCalldataEncodedTailSize(structTy)));

    auto dataLoc = structTy.getDataLocation();
    std::unique_ptr<StructEncodeMemberReader> reader;
    switch (dataLoc) {
    case sol::DataLocation::CallData:
      reader = std::make_unique<StructEncodeMemberReaderCallData>(
          structTy, *this, b, loc, src);
      break;
    case sol::DataLocation::Memory:
      reader = std::make_unique<StructEncodeMemberReaderMemory>(structTy, *this,
                                                                b, loc, src);
      break;
    case sol::DataLocation::Storage:
      reader = std::make_unique<StructEncodeMemberReaderStorage>(
          structTy, *this, b, loc, src);
      break;
    default:
      llvm_unreachable("Unexpected data location for struct encoding");
    }

    Value structHeadAddr = dstAddr;
    auto memberTypes = structTy.getMemberTypes();
    for (uint64_t i = 0, e = memberTypes.size(); i < e; ++i) {
      Type memTy = memberTypes[i];
      Value srcVal = reader->read(i);

      if (sol::hasDynamicallySizedElt(memTy)) {
        // Dynamic members store a tail offset in the head, then encode payload
        // at the current tail.
        b.create<yul::MStoreOp>(
            loc, structHeadAddr,
            b.create<arith::SubIOp>(loc, tailAddr, dstAddr));
        tailAddr =
            genABITupleEncoding(memTy, srcVal, tailAddr, /*dstAddrInTail=*/true,
                                dstAddr, tailAddr, loc, dataLoc);
      } else {
        // Static members never advance the tail on their own.
        tailAddr = genABITupleEncoding(memTy, srcVal, structHeadAddr,
                                       /*dstAddrInTail=*/false, dstAddr,
                                       tailAddr, loc, dataLoc);
      }

      // Advance the addresses iff there are more members to encode.
      if (i + 1 < e) {
        structHeadAddr = b.create<arith::AddIOp>(
            loc, structHeadAddr, bExt.genI256Const(getCallDataHeadSize(memTy)));
        reader->advance(i);
      }
    }
    return tailAddr;
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

  // External function ref.
  if (isa<sol::ExtFuncRefType>(ty)) {
    Value normalized = normalizeABIScalarForEncoding(ty, val, loc);
    b.create<yul::MStoreOp>(loc, addr, normalized);
    return b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(24));
  }

  // String type.
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    Value dataLen = genCopyStringDataToMemory(val, ty, addr, loc);
    return b.create<arith::AddIOp>(loc, addr, dataLen);
  }

  // Array type.
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    auto isValidPackedArrayElementType = [&](Type eltTy) {
      while (auto nestedArrTy = dyn_cast<sol::ArrayType>(eltTy)) {
        if (nestedArrTy.isDynSized())
          return false;
        eltTy = nestedArrTy.getEltType();
      }
      return isa<IntegerType>(eltTy) || isa<sol::EnumType>(eltTy) ||
             isa<sol::BytesType>(eltTy) || isa<sol::ExtFuncRefType>(eltTy) ||
             sol::isAddressLikeType(eltTy);
    };

    // TODO: Move packed array element type validation to a verifier.
    if (!isValidPackedArrayElementType(arrTy.getEltType()))
      llvm_unreachable(
          "Only scalar types and fixed nested arrays can be packed");

    return genABITupleEncoding(
        arrTy, val, addr, /*dstAddrInTail=*/true, /*tupleStart=*/addr,
        /*tailAddr=*/addr, loc, /*srcDataLoc=*/std::nullopt,
        /*includeLengthPrefix=*/false);
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

  // Revert if reading one ABI word (32 bytes) at 'addr' would exceed tuple
  // bounds.
  auto genRevertIfTupleWordOutOfBounds = [&](std::string const &revertMsg) {
    auto invalidRangeCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge,
        b.create<arith::AddIOp>(loc, addr, bExt.genI256Const(31)), tupleEnd);
    genRevertWithMsg(invalidRangeCond, revertMsg, loc);
  };

  // Revert if offset is above uint64 max (0xffffffffffffffff).
  auto genRevertIfOffsetTooLarge = [&](Value offset,
                                       std::string const &revertMsg) {
    auto invalidOffsetCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, offset,
        bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    genRevertWithMsg(invalidOffsetCond, revertMsg, loc);
  };

  // Revert if a value is past tuple end.
  auto genRevertIfPastTupleEnd = [&](Value value,
                                     std::string const &revertMsg) {
    auto invalidRangeCond = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, value, tupleEnd);
    genRevertWithMsg(invalidRangeCond, revertMsg, loc);
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

    genRevertIfTupleWordOutOfBounds(
        "ABI decoding: invalid calldata array offset");

    Value dstAddr, srcAddr, size, ret;
    Value thirtyTwo = bExt.genI256Const(32);
    if (arrTy.isDynSized()) {
      Value i256Size = genLoad(addr);
      srcAddr = b.create<arith::AddIOp>(loc, addr, thirtyTwo);

      // Generate an assertion that checks the size.
      auto scaledSize = b.create<arith::MulIOp>(
          loc, i256Size,
          bExt.genI256Const(getCallDataHeadSize(arrTy.getEltType())));
      auto endAddr = b.create<arith::AddIOp>(loc, srcAddr, scaledSize);
      genRevertIfPastTupleEnd(endAddr,
                              "ABI decoding: invalid calldata array stride");

      if (arrTy.getDataLocation() == sol::DataLocation::CallData)
        return bExt.genLLVMStruct({srcAddr, i256Size});

      dstAddr = genMemAllocForDynArray(
          i256Size, b.create<arith::MulIOp>(loc, i256Size, thirtyTwo), loc,
          true);
      ret = dstAddr;
      // Skip the size fields in both the addresses.
      dstAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);
      size = bExt.genCastToIdx(i256Size);
    } else {
      dstAddr = genMemAlloc(bExt.genI256Const(arrTy.getSize() * 32), loc);
      ret = dstAddr;
      srcAddr = addr;
      size = bExt.genIdxConst(arrTy.getSize());

      auto fixedSize =
          arrTy.getSize() * getCallDataHeadSize(arrTy.getEltType());
      Value srcEnd =
          b.create<arith::AddIOp>(loc, srcAddr, bExt.genI256Const(fixedSize));
      genRevertIfPastTupleEnd(srcEnd,
                              "ABI decoding: invalid calldata array stride");
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
            Value innerOffset = genLoad(iSrcAddr);
            genRevertIfOffsetTooLarge(
                innerOffset, "ABI decoding: invalid calldata array offset");

            // The elements are offset wrt to the start of this array (after the
            // size field if dynamic) that contain the inner element.
            Value offsetFromSrcArr =
                b.create<arith::AddIOp>(loc, srcAddr, innerOffset);
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

  // External function ref
  if (isa<sol::ExtFuncRefType>(ty)) {
    // Validate low 64 bits are zero.
    Value arg = genLoad(addr);
    APInt mask = APInt::getHighBitsSet(256, 192);
    Value maskedArg =
        b.create<arith::AndIOp>(loc, arg, bExt.genI256Const(mask));
    auto revertCond =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, arg, maskedArg);
    genRevert(revertCond, loc);
    return arg;
  }

  // String type
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    genRevertIfTupleWordOutOfBounds(
        "ABI decoding: invalid calldata array offset");

    Value tailAddr = addr;
    Value sizeInBytes = genLoad(tailAddr);
    Value thirtyTwo = bExt.genI256Const(32);
    Value srcDataAddr = b.create<arith::AddIOp>(loc, tailAddr, thirtyTwo);
    Value endAddr = b.create<arith::AddIOp>(loc, srcDataAddr, sizeInBytes);
    genRevertIfPastTupleEnd(endAddr, "ABI decoding: invalid byte array length");

    if (stringTy.getDataLocation() == sol::DataLocation::CallData)
      return bExt.genLLVMStruct({srcDataAddr, sizeInBytes});

    // Copy the decoded string to a new memory allocation.
    Value dstAddr = genMemAllocForDynArray(
        sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes), loc, true);
    Value dstDataAddr = b.create<arith::AddIOp>(loc, dstAddr, thirtyTwo);

    if (fromMem)
      // TODO? Check m_evmVersion.hasMcopy() and legalize here or in sol.mcopy
      // lowering?
      b.create<yul::MCopyOp>(loc, dstDataAddr, srcDataAddr, sizeInBytes);
    else
      b.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr, sizeInBytes);

    // Canonicalize the trailing bytes after a variable-length copy.
    // This clears the 32-byte word starting at dst + length. As in the
    // existing decode helper flow, this write may extend past the
    // rounded payload boundary.
    Value cleanupAddr = b.create<arith::AddIOp>(loc, dstDataAddr, sizeInBytes);
    b.create<yul::MStoreOp>(loc, cleanupAddr, bExt.genI256Const(0));

    return dstAddr;
  }

  // Struct type
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    auto dataLoc = structTy.getDataLocation();
    assert((dataLoc == sol::DataLocation::CallData ||
            dataLoc == sol::DataLocation::Memory) &&
           "Unexpected struct data location");

    std::string const &headTooShortMsg =
        dataLoc == sol::DataLocation::CallData
            ? "ABI decoding: struct calldata too short"
            : "ABI decoding: struct data too short";
    Value srcEnd = b.create<arith::AddIOp>(
        loc, addr,
        bExt.genI256Const(getStructCalldataEncodedTailSize(structTy)));
    genRevertIfPastTupleEnd(srcEnd, headTooShortMsg);

    // Keep calldata structs as calldata pointers.
    if (dataLoc == sol::DataLocation::CallData)
      return addr;

    Value dstAddr = genMemAlloc(
        bExt.genI256Const(structTy.getMemberTypes().size() * 32), loc);
    Value dstHeadAddr = dstAddr;
    Value srcHeadAddr = addr;
    for (Type memTy : structTy.getMemberTypes()) {
      Value memberVal;
      if (sol::hasDynamicallySizedElt(memTy)) {
        Value tailOffset = genLoad(srcHeadAddr);
        genRevertIfOffsetTooLarge(tailOffset,
                                  "ABI decoding: invalid struct offset");

        Value tailAddr = b.create<arith::AddIOp>(loc, addr, tailOffset);
        memberVal =
            genABITupleDecoding(memTy, tailAddr, fromMem, addr, tupleEnd, loc);
      } else {
        memberVal = genABITupleDecoding(memTy, srcHeadAddr, fromMem, addr,
                                        tupleEnd, loc);
      }

      // If the element type is an integer smaller than 256 bits, we need
      // to extend it. In case we don't do that, store will place the
      // value in the higher bits, which is incorrect.
      auto intTy = dyn_cast<IntegerType>(memberVal.getType());
      if (intTy && intTy.getWidth() < 256)
        memberVal =
            bExt.genIntCast(/*width=*/256, intTy.isSigned(), memberVal, loc);

      b.create<yul::MStoreOp>(loc, dstHeadAddr, memberVal);

      srcHeadAddr = b.create<arith::AddIOp>(
          loc, srcHeadAddr, bExt.genI256Const(getCallDataHeadSize(memTy)));
      dstHeadAddr =
          b.create<arith::AddIOp>(loc, dstHeadAddr, bExt.genI256Const(32));
    }

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
      Value tailOffset = genLoad(headAddr);
      auto invalidOffsetCond = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, tailOffset,
          bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
      genRevertWithMsg(invalidOffsetCond, "ABI decoding: invalid tuple offset",
                       loc);

      Value tailAddr = b.create<arith::AddIOp>(loc, tupleStart, tailOffset);
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

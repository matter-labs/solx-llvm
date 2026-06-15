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

#include "mlir/Conversion/SolToYul/EVMUtil.h"
#include "mlir/Conversion/SolToYul/EVMConstants.h"
#include "mlir/Conversion/SolToYul/Util.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "llvm/ADT/SmallSet.h"
#include <functional>

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

struct StructEncodeMemberReaderCallData final : evm::StructEncodeMemberReader {
  // Base address of this struct head in calldata.
  Value baseAddr;
  // Current calldata head cursor for member traversal.
  Value curAddr;

  StructEncodeMemberReaderCallData(sol::StructType structTy, evm::Builder &evmB,
                                   OpBuilder &b, Location loc, Value baseAddr)
      : evm::StructEncodeMemberReader(structTy, evmB, b, loc),
        baseAddr(baseAddr), curAddr(baseAddr) {}

  Value read(uint64_t memberIdx, bool /*skipCleanup*/ = false) override {
    Type memTy = getMemberType(memberIdx);
    if (sol::hasDynamicallySizedElt(memTy))
      return evmB.genCalldataAccessRef(memTy, baseAddr, curAddr,
                                       /*isNonABI=*/false, loc);

    if (sol::isNonPtrRefType(memTy))
      // Static non-pointer refs are laid out inline in calldata head and
      // should be passed by address.
      return curAddr;

    return evmB.genLoad(curAddr, sol::DataLocation::CallData, loc);
  }

  void advance(uint64_t memberIdx) override {
    Type memTy = getMemberType(memberIdx);
    curAddr = b.create<yul::AddOp>(
        loc, curAddr, bExt.genI256Const(evm::getCallDataHeadSize(memTy)));
  }
};

struct StructEncodeMemberReaderMemory final : evm::StructEncodeMemberReader {
  // Current memory head cursor for member traversal.
  Value curAddr;

  StructEncodeMemberReaderMemory(sol::StructType structTy, evm::Builder &evmB,
                                 OpBuilder &b, Location loc, Value curAddr)
      : evm::StructEncodeMemberReader(structTy, evmB, b, loc),
        curAddr(curAddr) {}

  Value read(uint64_t, bool /*skipCleanup*/ = false) override {
    Value srcVal = evmB.genLoad(curAddr, sol::DataLocation::Memory, loc);
    return srcVal;
  }

  void advance(uint64_t) override {
    // In memory, each struct member head entry occupies one 32-byte slot.
    curAddr = b.create<yul::AddOp>(loc, curAddr, bExt.genI256Const(32));
  }
};

struct StructEncodeMemberReaderStorage final : evm::StructEncodeMemberReader {
  // Base storage slot for the struct.
  Value baseAddr;

  // Cache the last loaded storage slot to avoid repeated sload for packed
  // members that share the same slot.
  std::optional<uint64_t> previousSlotOffset;
  Value previousSlotValue;

  StructEncodeMemberReaderStorage(sol::StructType structTy, evm::Builder &evmB,
                                  OpBuilder &b, Location loc, Value baseAddr)
      : evm::StructEncodeMemberReader(structTy, evmB, b, loc),
        baseAddr(baseAddr) {}

  Value read(uint64_t memberIdx, bool skipCleanup = false) override {
    auto [slotOffset, byteOffset] = structTy.getStorageMemberOffset(memberIdx);
    Type memTy = getMemberType(memberIdx);
    if (sol::canBePacked(memTy)) {
      // Packed members are extracted from the resolved slot and cleaned to the
      // source type width.
      if (!previousSlotOffset || *previousSlotOffset != slotOffset) {
        Value memberSlot =
            b.create<yul::AddOp>(loc, baseAddr, bExt.genI256Const(slotOffset));
        previousSlotValue =
            evmB.genLoad(memberSlot, sol::DataLocation::Storage, loc);
        previousSlotOffset = slotOffset;
      }

      Value shifted = previousSlotValue;
      if (byteOffset != 0)
        shifted = b.create<yul::ArithShrOp>(loc, previousSlotValue,
                                            bExt.genI256Const(byteOffset * 8));

      if (skipCleanup)
        return shifted;
      return evmB.genCleanupPackedStorageValue(memTy, shifted, loc);
    }

    // For a ref type, return its address.
    return b.create<yul::AddOp>(loc, baseAddr, bExt.genI256Const(slotOffset));
  }

  // Storage traversal is indexed by member number and precomputed offsets.
  void advance(uint64_t) override {}
};

struct StructMemberWriterStorage final : evm::StructMemberWriter {
  // Base storage slot for the struct.
  Value baseAddr;

  // Cache the last storage slot accumulator and its bitmask to avoid repeated
  // sstore for packed members that share the same slot.
  Value accum;
  // Compile-time OR of each field's bit range within the current slot.
  // Bits set here are owned by the new value. The rest are preserved from the
  // existing slot contents via a read-modify-write at flush time.
  APInt accumMask;
  std::optional<uint64_t> previousSlotOffset;

  StructMemberWriterStorage(sol::StructType structTy, evm::Builder &evmB,
                            OpBuilder &b, Location loc, Value baseAddr)
      : evm::StructMemberWriter(structTy, evmB, b, loc), baseAddr(baseAddr),
        accum(nullptr), accumMask(APInt::getZero(256)) {}

  void write(uint64_t memberIdx, Value srcVal, Type srcMemberTy) override {
    auto [slotOffset, byteOffset] = structTy.getStorageMemberOffset(memberIdx);
    Type dstMemberTy = getMemberType(memberIdx);
    if (sol::canBePacked(dstMemberTy)) {
      if (!previousSlotOffset || *previousSlotOffset != slotOffset) {
        accum = bExt.genI256Const(0);
        accumMask = APInt::getZero(256);
        previousSlotOffset = slotOffset;
      }

      // srcVal arrives pre-cleaned by the struct copy loop: it holds the raw
      // right-shifted field bits, already masked to the field width by
      // AND(fieldMask). The writer just positions and accumulates the bits.
      unsigned storageBits = sol::getNumBytes(dstMemberTy) * 8;
      APInt fieldMask = APInt::getLowBitsSet(256, storageBits);
      Value shiftBits = bExt.genI256Const(byteOffset * 8);
      Value shifted = byteOffset > 0
                          ? b.create<yul::ArithShlOp>(loc, srcVal, shiftBits)
                          : srcVal;
      accum = b.create<yul::OrOp>(loc, accum, shifted);

      // Track which bits of the slot belong to this field so that the flush
      // can preserve whatever lives in the remaining bits.
      fieldMask <<= (byteOffset * 8);
      accumMask |= fieldMask;

      // Flush the accumulator once we've seen every packed member that shares
      // this slot: either this was the last member, or the next member lives
      // in a different slot. Deferring the sstore until the slot boundary
      // avoids writing a partially-filled word and then overwriting it in the
      // very next iteration.
      ArrayRef<uint64_t> slotOffsets = structTy.getMemberSlotOffsets();
      uint64_t nextMemberIdx = memberIdx + 1;
      if ((nextMemberIdx == slotOffsets.size()) ||
          (slotOffsets[nextMemberIdx] != slotOffset)) {
        Value dstSlot =
            b.create<yul::AddOp>(loc, baseAddr, bExt.genI256Const(slotOffset));
        APInt fullSlotMask = APInt::getAllOnes(256);
        if (accumMask == fullSlotMask) {
          // We are going to overwrite the whole slot, so no need to read
          // the initial value of the slot.
          evmB.genStore(accum, dstSlot, sol::DataLocation::Storage, loc);
        } else {
          Value oldSlot =
              evmB.genLoad(dstSlot, sol::DataLocation::Storage, loc);
          Value preserved =
              b.create<yul::AndOp>(loc, oldSlot, bExt.genI256Const(~accumMask));
          Value newSlot = b.create<yul::OrOp>(loc, preserved, accum);
          evmB.genStore(newSlot, dstSlot, sol::DataLocation::Storage, loc);
        }
      }
    } else {
      // Only aggregates reach here, all are ref types whose readers return
      // an address from the source location.
      assert(sol::isNonPtrRefType(dstMemberTy));
      Value dstAddr =
          b.create<yul::AddOp>(loc, baseAddr, bExt.genI256Const(slotOffset));
      evmB.genCopy(srcMemberTy, dstMemberTy, srcVal, dstAddr,
                   sol::getDataLocation(srcMemberTy),
                   sol::DataLocation::Storage, loc);
    }
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
  auto forOp = bExt.createCountedLoop(
      bExt.genI256Const(0), size, bExt.genI256Const(1),
      ValueRange{dstStart, srcStart},
      [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
        Value iDstAddr = initArgs[0];
        Value iSrcAddr = initArgs[1];

        emitElement(b, loc, iSrcAddr, iDstAddr);

        Value nextDstAddr = b.create<yul::AddOp>(loc, iDstAddr, dstStride);
        Value nextSrcAddr = b.create<yul::AddOp>(loc, iSrcAddr, srcStride);
        return SmallVector<Value>{nextDstAddr, nextSrcAddr};
      });
  return forOp.getResult(1);
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
Value emitCompactStorageArrayReadLoop(OpBuilder &b, Location loc, Value size,
                                      Value dstStart, Value srcStart,
                                      Value dstStride, unsigned eltByteSize,
                                      bool isDynSized,
                                      EmitElementFromSlotFuncT &&emitElement) {
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned itemsPerSlot = 32 / eltByteSize;
  assert(itemsPerSlot > 1 && "Expected at least two packed items per slot");

  // Reused constants.
  Value zeroI256 = bExt.genI256Const(0);
  Value itemsPerSlotI256 = bExt.genI256Const(itemsPerSlot);

  // TODO: If we see some perf regressions, we can change this computation, as
  // LLVM will produce llvm.usub.sat.i256 for dynamic arrays which is
  // not efficient.
  auto hasFullSlots =
      bExt.genCmp(yul::CmpPredicate::uge, size, itemsPerSlotI256);
  Value fullLoopUpperBoundI256 = b.create<yul::ArithSelectOp>(
      loc, hasFullSlots,
      b.create<yul::SubOp>(loc, size, bExt.genI256Const(itemsPerSlot - 1)),
      zeroI256);

  // Full-slot loop.
  auto fullSlotForOp = bExt.createCountedLoop(
      zeroI256, fullLoopUpperBoundI256, itemsPerSlotI256,
      ValueRange{dstStart, srcStart, zeroI256},
      [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
        Value iDstAddr = initArgs[0];
        Value iSrcSlot = initArgs[1];
        Value iItemCounter = initArgs[2];

        // Load the slot once and extract all packed elements from it.
        Value slotVal = b.create<yul::SLoadOp>(loc, iSrcSlot);
        auto perSlotForOp = bExt.createCountedLoop(
            zeroI256, itemsPerSlotI256, bExt.genI256Const(1),
            ValueRange{iDstAddr, zeroI256},
            [&](OpBuilder &b, Location loc, Value, ValueRange perSlotArgs) {
              Value jDstAddr = perSlotArgs[0];
              Value jShiftBits = perSlotArgs[1];

              emitElement(b, loc, slotVal, jShiftBits, jDstAddr);

              // Advance destination and extraction offset for next item
              // in this same slot.
              Value nextDstAddr =
                  b.create<yul::AddOp>(loc, jDstAddr, dstStride);
              Value nextShiftBits = b.create<yul::AddOp>(
                  loc, jShiftBits, bExt.genI256Const(eltByteSize * 8));
              return SmallVector<Value>{nextDstAddr, nextShiftBits};
            },
            // Full unroll is beneficial for small itemsPerSlot counts, but for
            // large counts (uint8: 32/slot, uint16: 16/slot) it creates too
            // many simultaneously live values and overflows the EVM spill stack
            // region.
            // TODO: #59, tune loop unroll threshold for per-slot packing loops.
            /*fullUnroll=*/itemsPerSlot <= 8);

        // After consuming one full slot, move to the next storage slot
        // and advance the emitted item counter by itemsPerSlot.
        Value nextSrcSlot =
            b.create<yul::AddOp>(loc, iSrcSlot, bExt.genI256Const(1));
        Value nextItemCounter =
            b.create<yul::AddOp>(loc, iItemCounter, itemsPerSlotI256);
        return SmallVector<Value>{perSlotForOp.getResult(1), nextSrcSlot,
                                  nextItemCounter};
      });

  Value dstAfterFullSlots = fullSlotForOp.getResult(1);
  Value srcSlotAfterFullSlots = fullSlotForOp.getResult(2);
  Value itemCounterAfterFullSlots = fullSlotForOp.getResult(3);

  // Remainder phase handles tail elements in a partially-filled slot, if any.
  // It reuses one final slot load and emits only the still-missing items.
  auto hasRem =
      bExt.genCmp(yul::CmpPredicate::ult, itemCounterAfterFullSlots, size);
  auto remIf = bExt.createIf(TypeRange{dstAfterFullSlots.getType()}, hasRem);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&remIf.getThenRegion().front());

    // Load exactly the slot that contains the remaining tail elements.
    Value slotVal = b.create<yul::SLoadOp>(loc, srcSlotAfterFullSlots);

    // Remainder loop.
    // TODO: For dynamic arrays, solc unrolls this loop manually with
    // conditions. If we see some perf regressions, we can do the same.
    auto remForOp = bExt.createCountedLoop(
        itemCounterAfterFullSlots, size, bExt.genI256Const(1),
        ValueRange{dstAfterFullSlots, zeroI256},
        [&](OpBuilder &b, Location loc, Value, ValueRange remArgs) {
          Value remDstAddr = remArgs[0];
          Value remShiftBits = remArgs[1];

          emitElement(b, loc, slotVal, remShiftBits, remDstAddr);

          // Advance to next output element and next packed field in the
          // same loaded remainder slot.
          Value nextDstAddr = b.create<yul::AddOp>(loc, remDstAddr, dstStride);
          Value nextShiftBits = b.create<yul::AddOp>(
              loc, remShiftBits, bExt.genI256Const(eltByteSize * 8));
          return SmallVector<Value>{nextDstAddr, nextShiftBits};
        },
        // TODO: #59, tune loop unroll threshold for per-slot packing loops.
        /*fullUnroll=*/!isDynSized);

    b.create<yul::YieldOp>(loc, remForOp.getResult(1));
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&remIf.getElseRegion().front());
    b.create<yul::YieldOp>(loc, dstAfterFullSlots);
  }

  return remIf.getResult(0);
}

// Packs elements from a linear non-storage source into packed storage slots.
// Pseudo C:
//   const size_t itemsPerSlot = 32 / eltByteSize;
//   const size_t fullLoopUpperBound =
//       (size >= itemsPerSlot) ? (size - (itemsPerSlot - 1)) : 0;
//   src = srcStart;
//   dstSlot = dstSlotStart;
//   itemCounter = 0;
//
//   // Full slot loop.
//   for (; itemCounter < fullLoopUpperBound; itemCounter += itemsPerSlot) {
//     slotVal = 0;
//     for (size_t j = 0, shiftBits = 0; j < itemsPerSlot;
//          ++j, shiftBits += eltByteSize * 8) {
//       auto [newSlot, nextSrc] = emitElement(src, slotVal, shiftBits);
//       slotVal = newSlot;
//       src = nextSrc;
//     }
//     sstore(dstSlot, slotVal);
//     ++dstSlot;
//   }
//
//   if (itemCounter < size) {  // partial last slot
//     slotVal = 0;
//     for (size_t remIdx = itemCounter, shiftBits = 0; remIdx < size;
//          ++remIdx, shiftBits += eltByteSize * 8) {
//       auto [newSlot, nextSrc] = emitElement(src, slotVal, shiftBits);
//       slotVal = newSlot;
//       src = nextSrc;
//     }
//     sstore(dstSlot, slotVal);
//   }
//
// emitElement(b, loc, srcAddr, accumulator, shiftBits) ->
//   {newAccumulator, nextSrcAddr}
template <typename EmitElementFuncT>
void emitCompactStorageArrayWriteLoop(OpBuilder &b, Location loc, Value size,
                                      Value dstSlotStart, Value srcStart,
                                      unsigned eltByteSize, bool isDynSized,
                                      EmitElementFuncT &&emitElement) {
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned itemsPerSlot = 32 / eltByteSize;
  assert(itemsPerSlot > 1 && "Expected at least two packed items per slot");

  Value zeroI256 = bExt.genI256Const(0);
  Value itemsPerSlotI256 = bExt.genI256Const(itemsPerSlot);

  auto hasFullSlots =
      bExt.genCmp(yul::CmpPredicate::uge, size, itemsPerSlotI256);
  Value fullLoopUpperBoundI256 = b.create<yul::ArithSelectOp>(
      loc, hasFullSlots,
      b.create<yul::SubOp>(loc, size, bExt.genI256Const(itemsPerSlot - 1)),
      zeroI256);

  // Full-slot outer loop: iter_args = {srcCursor, dstSlot, itemCounter}
  auto fullSlotForOp = bExt.createCountedLoop(
      zeroI256, fullLoopUpperBoundI256, itemsPerSlotI256,
      ValueRange{srcStart, dstSlotStart, zeroI256},
      [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
        Value iSrcCursor = initArgs[0];
        Value iDstSlot = initArgs[1];
        Value iItemCounter = initArgs[2];

        // Inner loop: accumulate itemsPerSlot elements into one slot value.
        // iter_args = {srcCursor, shiftBits, slotAccumulator}
        auto perSlotForOp = bExt.createCountedLoop(
            zeroI256, itemsPerSlotI256, bExt.genI256Const(1),
            ValueRange{iSrcCursor, zeroI256, zeroI256},
            [&](OpBuilder &b, Location loc, Value, ValueRange perSlotArgs) {
              Value jSrcAddr = perSlotArgs[0];
              Value jShiftBits = perSlotArgs[1];
              Value jAccum = perSlotArgs[2];

              auto [newAccum, nextSrcAddr] =
                  emitElement(b, loc, jSrcAddr, jAccum, jShiftBits);

              Value nextShiftBits = b.create<yul::AddOp>(
                  loc, jShiftBits, bExt.genI256Const(eltByteSize * 8));
              return SmallVector<Value>{nextSrcAddr, nextShiftBits, newAccum};
            },
            // TODO: #59, tune loop unroll threshold for per-slot packing loops.
            /*fullUnroll=*/itemsPerSlot <= 8);

        b.create<yul::SStoreOp>(loc, iDstSlot, perSlotForOp.getResult(3));

        Value nextDstSlot =
            b.create<yul::AddOp>(loc, iDstSlot, bExt.genI256Const(1));
        Value nextItemCounter =
            b.create<yul::AddOp>(loc, iItemCounter, itemsPerSlotI256);
        return SmallVector<Value>{perSlotForOp.getResult(1), nextDstSlot,
                                  nextItemCounter};
      });

  Value srcAfterFull = fullSlotForOp.getResult(1);
  Value dstSlotAfterFull = fullSlotForOp.getResult(2);
  Value itemCounterAfterFull = fullSlotForOp.getResult(3);

  // Remainder: handle a partial last storage slot, if any.
  auto hasRem = bExt.genCmp(yul::CmpPredicate::ult, itemCounterAfterFull, size);
  auto remIf = bExt.createIf(hasRem);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&remIf.getThenRegion().front());

    auto remForOp = bExt.createCountedLoop(
        itemCounterAfterFull, size, bExt.genI256Const(1),
        ValueRange{srcAfterFull, zeroI256, zeroI256},
        [&](OpBuilder &b, Location loc, Value, ValueRange remArgs) {
          Value remSrcAddr = remArgs[0];
          Value remShiftBits = remArgs[1];
          Value remAccum = remArgs[2];

          auto [newAccum, nextSrcAddr] =
              emitElement(b, loc, remSrcAddr, remAccum, remShiftBits);

          Value nextShiftBits = b.create<yul::AddOp>(
              loc, remShiftBits, bExt.genI256Const(eltByteSize * 8));
          return SmallVector<Value>{nextSrcAddr, nextShiftBits, newAccum};
        },
        // TODO: #59, tune loop unroll threshold for per-slot packing loops.
        /*fullUnroll=*/!isDynSized);
    b.create<yul::SStoreOp>(loc, dstSlotAfterFull, remForOp.getResult(3));
  }
}

// Copies numItems packed items from source storage slots to destination
// storage slots, converting item sizes on the fly. Both src and dst use
// packed storage (byteSize <= 16). Their item sizes may differ.
//
// Pseudo C:
//   uint256 srcSlot = srcDataAddr, srcSlotVal = sload(srcDataAddr);
//   uint256 dstSlot = dstDataAddr, dstAccum = 0;
//   size_t  srcItem = 0, dstItem = 0;
//
//   for (size_t i = 0; i < numItems; ++i) {
//     uint256 val = (srcSlotVal >> (srcItem * srcEltByteSize*8)) & srcMask;
//
//     // [widen / narrow / mask for bytesN or integer conversion]
//
//     dstAccum |= dstVal << (dstItem * dstEltByteSize*8);
//     if (++srcItem == srcItemsPerSlot && i + 1 < numItems) {
//       srcSlot++;
//       srcSlotVal = sload(srcSlot);
//       srcItem = 0;
//     }
//
//     if (++dstItem == dstItemsPerSlot) {
//       sstore(dstSlot, dstAccum); dstSlot++; dstAccum = 0; dstItem = 0;
//     }
//   }
//
//   // flush partial last slot
//   if (dstItem > 0) sstore(dstSlot, dstAccum);
static void emitRepackStorageToStorageCopyLoop(OpBuilder &b, Location loc,
                                               Value numItems,
                                               Value srcDataAddr,
                                               Value dstDataAddr, Type srcEltTy,
                                               unsigned srcEltByteSize,
                                               unsigned dstEltByteSize) {
  mlir::solgen::BuilderExt bExt(b, loc);

  unsigned srcItemsPerSlot = 32 / srcEltByteSize;
  unsigned dstItemsPerSlot = 32 / dstEltByteSize;
  APInt srcMask = APInt::getLowBitsSet(256, srcEltByteSize * 8);
  APInt dstMask = APInt::getLowBitsSet(256, dstEltByteSize * 8);
  // BytesN values are big-endian packed: widening/narrowing requires a shift
  // to keep byte[0] at the most-significant position within the item's bit
  // field. Integer types are right-aligned and only need masking.
  auto i256Ty = b.getIntegerType(256);
  Value initSrcSlotVal = b.create<yul::SLoadOp>(loc, srcDataAddr);

  // iter_args: srcSlot(i256), srcSlotVal(i256), srcItemInSlot(index),
  //            dstSlot(i256), dstSlotAccum(i256), dstItemInSlot(index)
  auto loop = bExt.createCountedLoop(
      bExt.genI256Const(0), numItems, bExt.genI256Const(1),
      ValueRange{srcDataAddr, initSrcSlotVal, bExt.genI256Const(0), dstDataAddr,
                 bExt.genI256Const(0), bExt.genI256Const(0)},
      [&](OpBuilder &b, Location loc, Value loopIdx, ValueRange args) {
        Value srcSlot = args[0], srcSlotVal = args[1];
        Value srcItemInSlot = args[2];
        Value dstSlot = args[3], dstAccum = args[4];
        Value dstItemInSlot = args[5];

        // Extract src item (right-aligned, masked).
        Value srcShift = b.create<yul::MulOp>(
            loc, srcItemInSlot, bExt.genI256Const(srcEltByteSize * 8));
        Value raw = b.create<yul::ArithShrOp>(loc, srcSlotVal, srcShift);
        Value srcVal =
            b.create<yul::AndOp>(loc, raw, bExt.genI256Const(srcMask));

        // Convert to dst item type.
        Value dstVal = nullptr;
        if (isa<sol::FixedBytesType>(srcEltTy) &&
            srcEltByteSize < dstEltByteSize) {
          // Bytes widening: shift MSB toward the high end of dst field.
          dstVal = b.create<yul::ArithShlOp>(
              loc, srcVal,
              bExt.genI256Const((dstEltByteSize - srcEltByteSize) * 8));
        } else if (isa<sol::FixedBytesType>(srcEltTy) &&
                   srcEltByteSize > dstEltByteSize) {
          // Bytes narrowing: take the most-significant dst bytes.
          dstVal = b.create<yul::ArithShrOp>(
              loc, srcVal,
              bExt.genI256Const((srcEltByteSize - dstEltByteSize) * 8));
        } else {
          // Integer (or equal-width bytes): right-aligned, masking suffices.
          dstVal = srcVal;
        }

        dstVal = b.create<yul::AndOp>(loc, dstVal, bExt.genI256Const(dstMask));

        // Pack into current dst slot accumulator.
        Value dstShift = b.create<yul::MulOp>(
            loc, dstItemInSlot, bExt.genI256Const(dstEltByteSize * 8));
        Value packed = b.create<yul::ArithShlOp>(loc, dstVal, dstShift);
        Value newDstAccum = b.create<yul::OrOp>(loc, dstAccum, packed);

        // Advance source item index in slot. Reload slot on slot boundary,
        // but only when there are more items to process. The last iteration
        // never needs the next source slot.
        Value incSrc =
            b.create<yul::AddOp>(loc, srcItemInSlot, bExt.genI256Const(1));
        Value srcDone = bExt.genCmp(yul::CmpPredicate::eq, incSrc,
                                    bExt.genI256Const(srcItemsPerSlot));
        Value hasMore = bExt.genCmp(
            yul::CmpPredicate::ult,
            b.create<yul::AddOp>(loc, loopIdx, bExt.genI256Const(1)), numItems);
        Value shouldReload = b.create<yul::AndOp>(loc, srcDone, hasMore);
        auto srcIf = bExt.createIf(TypeRange{i256Ty, i256Ty, i256Ty},
                                   shouldReload, /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&srcIf.getThenRegion().front());
          Value nextSlot =
              b.create<yul::AddOp>(loc, srcSlot, bExt.genI256Const(1));
          Value nextSlotVal = b.create<yul::SLoadOp>(loc, nextSlot);
          b.create<yul::YieldOp>(
              loc, ValueRange{nextSlot, nextSlotVal, bExt.genI256Const(0)});
        }
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&srcIf.getElseRegion().front());
          b.create<yul::YieldOp>(loc, ValueRange{srcSlot, srcSlotVal, incSrc});
        }

        // Advance destination item index and flush completed slots.
        Value incDst =
            b.create<yul::AddOp>(loc, dstItemInSlot, bExt.genI256Const(1));
        Value dstDone = bExt.genCmp(yul::CmpPredicate::eq, incDst,
                                    bExt.genI256Const(dstItemsPerSlot));
        auto dstIf = bExt.createIf(TypeRange{i256Ty, i256Ty, i256Ty}, dstDone,
                                   /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&dstIf.getThenRegion().front());
          b.create<yul::SStoreOp>(loc, dstSlot, newDstAccum);
          Value nextDstSlot =
              b.create<yul::AddOp>(loc, dstSlot, bExt.genI256Const(1));
          b.create<yul::YieldOp>(loc,
                                 ValueRange{nextDstSlot, bExt.genI256Const(0),
                                            bExt.genI256Const(0)});
        }
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&dstIf.getElseRegion().front());
          b.create<yul::YieldOp>(loc, ValueRange{dstSlot, newDstAccum, incDst});
        }

        return SmallVector<Value>{srcIf.getResult(0), srcIf.getResult(1),
                                  srcIf.getResult(2), dstIf.getResult(0),
                                  dstIf.getResult(1), dstIf.getResult(2)};
      });

  // Flush any partial last dst slot.
  Value hasTail = bExt.genCmp(yul::CmpPredicate::ugt, loop.getResult(6),
                              bExt.genI256Const(0));
  auto tailIf = bExt.createIf(hasTail);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&tailIf.getThenRegion().front());
    b.create<yul::SStoreOp>(loc, loop.getResult(4), loop.getResult(5));
  }
}

// Returns true when a calldata array element type can be copied directly as a
// 32-byte word in the fast-path array copy.
bool canFastCopyCalldataArray(Type eltTy) {
  if (auto intTy = dyn_cast<IntegerType>(eltTy))
    // TODO: Can we allow signed integers here as well?
    return intTy.getWidth() == 256 && !intTy.isSigned();

  auto bytesTy = dyn_cast<sol::FixedBytesType>(eltTy);
  return bytesTy && bytesTy.getSize() == 32;
}

// Centralizes common ABI decode bounds checks and the canonical diagnostic
// messages used when those checks fail.
struct ABIDecodeGuards {
  static constexpr char const *kInvalidArrayOffset =
      "ABI decoding: invalid calldata array offset";
  static constexpr char const *kInvalidArrayLength =
      "ABI decoding: invalid calldata array length";
  static constexpr char const *kInvalidArrayStride =
      "ABI decoding: invalid calldata array stride";
  static constexpr char const *kInvalidByteArrayLength =
      "ABI decoding: invalid byte array length";
  static constexpr char const *kInvalidTupleOffset =
      "ABI decoding: invalid tuple offset";
  static constexpr char const *kInvalidStructOffset =
      "ABI decoding: invalid struct offset";
  static constexpr char const *kStructCalldataTooShort =
      "ABI decoding: struct calldata too short";
  static constexpr char const *kStructDataTooShort =
      "ABI decoding: struct data too short";

  evm::Builder &evmB;
  OpBuilder &b;
  Location loc;
  Value tupleEnd;
  mlir::solgen::BuilderExt bExt;

  ABIDecodeGuards(evm::Builder &evmB, OpBuilder &b, Location loc,
                  Value tupleEnd)
      : evmB(evmB), b(b), loc(loc), tupleEnd(tupleEnd), bExt(b, loc) {}

  // Ensures the length/offset word at a dynamic tail address can be loaded.
  void requireArrayOffsetInBounds(Value addr) {
    Value lastByte = b.create<yul::AddOp>(loc, addr, bExt.genI256Const(31));
    Value invalid = bExt.genCmp(yul::CmpPredicate::sge, lastByte, tupleEnd);
    evmB.genDebugRevertWithMsg(invalid, kInvalidArrayOffset, loc);
  }

  // ABI offsets and lengths are rejected when they exceed the uint64 range
  // used by the lowering and runtime helpers.
  void requireUint64(Value value, char const *msg) {
    Value invalid =
        bExt.genCmp(yul::CmpPredicate::ugt, value,
                    bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    evmB.genDebugRevertWithMsg(invalid, msg, loc);
  }

  // Validates that a computed end address stays within the tuple boundary.
  void requireEndInBounds(Value endAddr, char const *msg) {
    Value invalid = bExt.genCmp(yul::CmpPredicate::ugt, endAddr, tupleEnd);
    evmB.genDebugRevertWithMsg(invalid, msg, loc);
  }

  // Fixed arrays are ABI-laid out inline, so their full head span must fit.
  void requireFixedArraySpan(Value baseAddr, sol::ArrayType arrTy) {
    Value endAddr = b.create<yul::AddOp>(
        loc, baseAddr,
        bExt.genI256Const(arrTy.getSize() *
                          evm::getCallDataHeadSize(arrTy.getEltType())));
    requireEndInBounds(endAddr, kInvalidArrayStride);
  }

  // Dynamic arrays use element head stride to determine how much payload is
  // addressable from the start of the dynamic data region.
  void requireDynamicArraySpan(Value dataAddr, Value size, Type eltTy) {
    Value spanBytes = b.create<yul::MulOp>(
        loc, size, bExt.genI256Const(evm::getCallDataHeadSize(eltTy)));
    Value endAddr = b.create<yul::AddOp>(loc, dataAddr, spanBytes);
    requireEndInBounds(endAddr, kInvalidArrayStride);
  }

  // Struct head diagnostics differ only by whether the source is calldata or
  // memory, so the selection is centralized here.
  char const *getStructHeadTooShortMsg(sol::DataLocation dataLoc) const {
    return dataLoc == sol::DataLocation::CallData ? kStructCalldataTooShort
                                                  : kStructDataTooShort;
  }
};
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
      isa<sol::FixedBytesType>(ty) || isa<sol::ExtFuncRefType>(ty) ||
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

unsigned evm::getArrayEltStride(sol::ArrayType arrTy) {
  Type eltTy = arrTy.getEltType();
  switch (arrTy.getDataLocation()) {
  case sol::DataLocation::Memory:
    return 32;
  case sol::DataLocation::CallData:
    return getCallDataHeadSize(eltTy);
  case sol::DataLocation::Storage:
    return sol::getStorageSlotCount(eltTy);
  default:
    llvm_unreachable("Unexpected array data location");
  }
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

std::unique_ptr<evm::StructEncodeMemberReader>
evm::Builder::makeStructMemberReader(sol::StructType structTy, OpBuilder &b,
                                     Location loc, Value src) {
  switch (sol::getDataLocation(structTy)) {
  case sol::DataLocation::CallData:
    return std::make_unique<StructEncodeMemberReaderCallData>(structTy, *this,
                                                              b, loc, src);
  case sol::DataLocation::Memory:
    return std::make_unique<StructEncodeMemberReaderMemory>(structTy, *this, b,
                                                            loc, src);
  case sol::DataLocation::Storage:
    return std::make_unique<StructEncodeMemberReaderStorage>(structTy, *this, b,
                                                             loc, src);
  default:
    llvm_unreachable("Unexpected data location for struct member reader");
  }
}

std::unique_ptr<evm::StructMemberWriter>
evm::Builder::makeStructMemberWriter(sol::StructType structTy, OpBuilder &b,
                                     Location loc, Value dst) {
  switch (sol::getDataLocation(structTy)) {
  case sol::DataLocation::Storage:
    return std::make_unique<StructMemberWriterStorage>(structTy, *this, b, loc,
                                                       dst);
  default:
    llvm_unreachable("Unexpected data location for struct member writer");
  }
}

Value evm::Builder::genIntCleanup(unsigned width, bool isSigned, Value val,
                                  std::optional<Location> locArg) {
  [[maybe_unused]] auto srcType = cast<IntegerType>(val.getType());
  assert(srcType.isSignless());
  assert(width <= 256 && "Expected integer width <= 256");
  assert(srcType.getWidth() == 256 &&
         "Yul integer cleanup expects promoted i256 values");

  if (width == 256)
    return val;

  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // For bool, use `x != 0` semantics.
  if (width == 1)
    return bExt.genCmp(yul::CmpPredicate::ne, val, bExt.genI256Const(0, loc),
                       loc);

  if (!isSigned) {
    APInt mask = APInt::getLowBitsSet(256, width);
    return b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask, loc));
  }

  assert(width % 8 == 0 && "signed Yul cleanup expects byte-aligned widths");
  Value off = bExt.genI256Const((width / 8) - 1, loc);
  return b.create<yul::SignExtendOp>(loc, off, val);
}

Value evm::Builder::genCleanup(Type ty, Value val,
                               std::optional<Location> locArg,
                               std::optional<sol::DataLocation> srcDataLoc) {
  bool fromCalldata = srcDataLoc && *srcDataLoc == sol::DataLocation::CallData;
  return genCleanup(ty, val, locArg, /*shouldRevert=*/fromCalldata);
}

Value evm::Builder::genCleanup(Type ty, Value val,
                               std::optional<Location> locArg,
                               bool shouldRevert) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() == 256)
      return val;

    assert(intTy.getWidth() < 256 &&
           "Expected integer types no wider than 256 bits");

    Value cleaned = genIntCleanup(intTy.getWidth(), intTy.isSigned(), val, loc);
    if (shouldRevert) {
      Value revertCond = bExt.genCmp(yul::CmpPredicate::ne, val, cleaned);
      genRevert(revertCond, loc);
    }
    return cleaned;
  }

  if (auto enumTy = dyn_cast<sol::EnumType>(ty)) {
    Value revertCond = bExt.genCmp(yul::CmpPredicate::ugt, val,
                                   bExt.genI256Const(enumTy.getMax()));
    if (shouldRevert)
      genRevert(revertCond, loc);
    else
      genPanic(mlir::evm::PanicCode::EnumConversionError, revertCond, loc);
    return val;
  }

  if (sol::isAddressLikeType(ty)) {
    APInt mask = APInt::getLowBitsSet(256, 160);
    Value cleaned =
        b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask, loc));
    if (shouldRevert) {
      Value revertCond = bExt.genCmp(yul::CmpPredicate::ne, val, cleaned);
      genRevert(revertCond, loc);
    }
    return cleaned;
  }

  if (sol::isBytesLikeType(ty)) {
    unsigned byteSize = sol::getNumBytes(ty);
    if (byteSize == 32)
      return val;

    assert(byteSize < 32 && "Expected fixed-bytes width <= 32");
    APInt mask = APInt::getHighBitsSet(256, byteSize * 8);
    Value cleaned =
        b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask, loc));

    // For ByteType we should never generate reverts.
    if (!isa<sol::ByteType>(ty) && shouldRevert) {
      Value revertCond = bExt.genCmp(yul::CmpPredicate::ne, val, cleaned);
      genRevert(revertCond, loc);
    }
    return cleaned;
  }

  if (isa<sol::FuncRefType>(ty)) {
    APInt mask = APInt::getLowBitsSet(256, 64);
    return b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask, loc));
  }

  if (isa<sol::ExtFuncRefType>(ty)) {
    APInt mask = APInt::getHighBitsSet(256, 192);
    Value cleaned =
        b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask, loc));
    if (shouldRevert) {
      Value revertCond = bExt.genCmp(yul::CmpPredicate::ne, val, cleaned);
      genRevert(revertCond, loc);
    }
    return cleaned;
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

  Value newFreePtr =
      b.create<yul::AddOp>(loc, freePtr, bExt.genRoundUpToMultiple<32>(size));

  Value isTooLarge =
      bExt.genCmp(yul::CmpPredicate::ugt, newFreePtr,
                  bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
  Value overflowed = bExt.genCmp(yul::CmpPredicate::ult, newFreePtr, freePtr);
  Value panicCond = b.create<yul::OrOp>(loc, isTooLarge, overflowed);
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
    auto panicCond =
        bExt.genCmp(yul::CmpPredicate::ugt, sizeVar,
                    bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    genPanic(mlir::evm::PanicCode::ResourceError, panicCond, loc);
  }

  // dynSize is size + length-slot where length-slot's size is 32 bytes.
  auto dynSizeInBytes =
      b.create<yul::AddOp>(loc, sizeInBytes, bExt.genI256Const(32));
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
        sizeInBytes = b.create<yul::MulOp>(loc, sizeVar, bExt.genI256Const(32));
        memPtr = genMemAllocForDynArray(sizeVar, sizeInBytes, loc);
        dataPtr = b.create<yul::AddOp>(loc, memPtr, bExt.genI256Const(32));
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
          addr = b.create<yul::AddOp>(loc, addr, bExt.genI256Const(32));
        }
        return memPtr;
      }

      //
      // Store the offsets to the "inner" allocations.
      //
      // Generate the loop for the stores of offsets.

      // `size` should be a multiple of 32.
      bExt.createCountedLoop(
          bExt.genI256Const(0), sizeInBytes, bExt.genI256Const(32),
          ValueRange{},
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange) {
            Value incrMemPtr = b.create<yul::AddOp>(loc, dataPtr, indVar);
            b.create<yul::MStoreOp>(
                loc, incrMemPtr,
                genMemAlloc(eltTy, zeroInit, initVals, sizeVar, recDepth, loc));
            return SmallVector<Value>{};
          });

    } else if (zeroInit) {
      Value callDataSz = b.create<yul::CallDataSizeOp>(loc);
      b.create<yul::CallDataCopyOp>(loc, dataPtr, callDataSz, sizeInBytes);

    } else {
      Value addr = dataPtr;
      for (auto val : initVals) {
        // Clean values before storing full memory words; array literals can
        // carry dirty high bits in narrow element values.
        b.create<yul::MStoreOp>(loc, addr, genCleanup(eltTy, val, loc));
        addr = b.create<yul::AddOp>(loc, addr, bExt.genI256Const(32));
      }
    }

    return memPtr;
  }

  // String type.
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    // CallData strings are {offset, length} references.
    // The zero/null value is {0, 0}.
    if (stringTy.getDataLocation() == sol::DataLocation::CallData)
      return bExt.genLLVMStruct({bExt.genI256Const(0), bExt.genI256Const(0)},
                                loc);
    if (sizeVar) {
      Value roundedSize = bExt.genRoundUpToMultiple<32>(sizeVar);
      Value memPtr = genMemAllocForDynArray(sizeVar, roundedSize, loc);
      if (zeroInit) {
        Value dataPtr =
            b.create<yul::AddOp>(loc, memPtr, bExt.genI256Const(32));
        Value callDataSz = b.create<yul::CallDataSizeOp>(loc);
        b.create<yul::CallDataCopyOp>(loc, dataPtr, callDataSz, roundedSize);
      }
      return memPtr;
    }
    return bExt.genI256Const(mlir::evm::MemoryLayout::zeroPointer);
  }

  // Struct type.
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    Value basePtr = genMemAlloc(evm::getMallocSize(ty), loc);
    assert(structTy.getDataLocation() == sol::DataLocation::Memory);

    uint64_t memOffset = 0;
    for (auto memTy : structTy.getMemberTypes()) {
      Value memPtr = memOffset == 0
                         ? basePtr
                         : b.create<yul::AddOp>(loc, basePtr,
                                                bExt.genI256Const(memOffset));
      if (isa<sol::StructType>(memTy) || isa<sol::ArrayType>(memTy)) {
        Value initVal =
            genMemAlloc(memTy, zeroInit, {}, sizeVar, recDepth, loc);
        b.create<yul::MStoreOp>(loc, memPtr, initVal);
      } else if (zeroInit) {
        // String/bytes fields must point to the zero-bytes sentinel (0x60) so
        // that dereferencing them yields length=0. When !zeroInit, these slots
        // are left at 0 (null pointer). This is safe only if the caller writes
        // every StringType field before any read. A future !zeroInit allocation
        // that skips a field would silently produce a null pointer dereference.
        Value zeroVal =
            isa<sol::StringType>(memTy)
                ? bExt.genI256Const(mlir::evm::MemoryLayout::zeroPointer)
                : bExt.genI256Const(0);
        b.create<yul::MStoreOp>(loc, memPtr, zeroVal);
      }
      memOffset += 32;
    }
    return basePtr;
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genMemAlloc(Type ty, bool zeroInit, ValueRange initVals,
                                Value sizeVar, Type sizeVarTy,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // If sizeVar and sizeVarTy are provided, insert cleanup for sizeVar. This is
  // necessary as sizeVar may contain dirty high bits.
  if (sizeVar && sizeVarTy)
    sizeVar = genCleanup(sizeVarTy, sizeVar, loc);

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
    return b.create<yul::AddOp>(loc, addr, bExt.genI256Const(32));
  }

  if (dataLoc == sol::DataLocation::Storage) {
    // Return the keccak256 of addr.
    auto zero = bExt.genI256Const(0);
    b.create<yul::MStoreOp>(loc, zero, addr);
    return b.create<yul::Keccak256Op>(loc, zero, bExt.genI256Const(32));
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genDynBytesToFixedBytes(Value src, Type srcTy,
                                            sol::FixedBytesType dstTy,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto srcStringTy = cast<sol::StringType>(srcTy);
  sol::DataLocation srcDataLoc = srcStringTy.getDataLocation();
  unsigned dstBytes = dstTy.getSize();

  llvm::APInt fixedMask = llvm::APInt::getAllOnes(256);
  if (dstBytes < 32)
    fixedMask = fixedMask.shl(256 - dstBytes * 8);
  Value fixedMaskVal = bExt.genI256Const(fixedMask, loc);

  Value length = nullptr;
  Value loadedWord = nullptr;
  if (srcDataLoc == sol::DataLocation::Storage) {
    Value lengthSlot = genLoad(src, srcDataLoc, loc);
    length = genStorageStringLength(lengthSlot, loc);

    // If length > 31, the payload lives in the out-of-place data area and the
    // first word must be loaded from there. Otherwise the payload is packed
    // into the slot word, so the slot contents already hold the first word.
    Value isOutOfPlace =
        bExt.genCmp(yul::CmpPredicate::ugt, length, bExt.genI256Const(31, loc));
    auto ifOutOfPlace = bExt.createIf(fixedMaskVal.getType(), isOutOfPlace);

    b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());
    b.create<yul::YieldOp>(
        loc, genLoad(genDataAddrPtr(src, srcTy, loc), srcDataLoc, loc));

    b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
    b.create<yul::YieldOp>(loc, lengthSlot);

    b.setInsertionPointAfter(ifOutOfPlace);
    loadedWord = ifOutOfPlace.getResult(0);
  } else {
    // Memory and calldata keep the payload contiguous after the header, so the
    // first word is a direct load from the data pointer.
    loadedWord = genLoad(genDataAddrPtr(src, srcTy, loc), srcDataLoc, loc);
    length = genDynSize(src, srcTy, loc);
  }

  // Fixed-bytes values occupy the high-order bytes of the 256-bit word. First
  // keep the destination-width prefix, then clear any bytes that lie past the
  // source's logical length.
  Value value = b.create<yul::AndOp>(loc, loadedWord, fixedMaskVal);

  // If length < dstBytes, the extracted prefix still contains bytes beyond the
  // logical end of the source, so clear that suffix. Otherwise the prefix is
  // already the final result.
  Value needsShortMask = bExt.genCmp(yul::CmpPredicate::ult, length,
                                     bExt.genI256Const(dstBytes, loc));
  auto ifNeedsShortMask = bExt.createIf(value.getType(), needsShortMask);

  b.setInsertionPointToStart(&ifNeedsShortMask.getThenRegion().front());

  // shiftBits = (dstBytes - length) * 8
  Value shiftBits = b.create<yul::MulOp>(
      loc, b.create<yul::SubOp>(loc, bExt.genI256Const(dstBytes, loc), length),
      bExt.genI256Const(8, loc));

  Value shortMask = nullptr;
  if (dstBytes == 32)
    // shiftBits can be 256 when length is 0, so use yul::shl
    // instead of the arithmetic wrapper, as the latter can produce poison
    // value in that case.
    shortMask = b.create<yul::ShlOp>(loc, shiftBits, fixedMaskVal);
  else
    // For smaller widths, it is safe to use the arithmetic wrapper, as
    // shiftBits won't be 256.
    shortMask = b.create<yul::ArithShlOp>(loc, fixedMaskVal, shiftBits);

  Value shortValue = b.create<yul::AndOp>(loc, value, shortMask);
  b.create<yul::YieldOp>(loc, shortValue);

  b.setInsertionPointToStart(&ifNeedsShortMask.getElseRegion().front());
  b.create<yul::YieldOp>(loc, value);

  b.setInsertionPointAfter(ifNeedsShortMask);
  return ifNeedsShortMask.getResult(0);
}

Value evm::Builder::genAddrAtIdx(Value baseAddr, Value idx, Type ty,
                                 sol::DataLocation dataLoc,
                                 std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  if (dataLoc == sol::DataLocation::Memory) {
    Value memIdx = b.create<yul::MulOp>(loc, idx, bExt.genI256Const(32));
    return b.create<yul::AddOp>(loc, baseAddr, memIdx);
  }

  if (dataLoc == sol::DataLocation::CallData) {
    unsigned stride = 32;
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
      stride = getCallDataHeadSize(arrTy.getEltType());
    Value memIdx = b.create<yul::MulOp>(loc, idx, bExt.genI256Const(stride));
    return b.create<yul::AddOp>(loc, baseAddr, memIdx);
  }

  if (dataLoc == sol::DataLocation::Storage) {
    Value stride;
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty))
      stride = bExt.genI256Const(sol::getStorageSlotCount(arrTy.getEltType()));
    else if (isa<sol::StringType>(ty))
      stride = bExt.genI256Const(1);
    Value scaledIdx = b.create<yul::MulOp>(loc, idx, stride);
    return b.create<yul::AddOp>(loc, baseAddr, scaledIdx);
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genCalldataAccessRef(Type ty, Value baseAddr, Value ptr,
                                         bool isNonABI,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  assert(sol::getDataLocation(ty) == sol::DataLocation::CallData &&
         "Expected calldata reference type");
  assert(sol::hasDynamicallySizedElt(ty) &&
         "Expected dynamically encoded calldata reference");

  const char *msgOffset = isNonABI ? "Invalid calldata tail offset"
                                   : "Invalid calldata access offset";
  const char *msgLength = isNonABI ? "Invalid calldata tail length"
                                   : "Invalid calldata access length";
  const char *msgStride =
      isNonABI ? "Calldata tail too short" : "Invalid calldata access stride";

  // Dynamically encoded calldata references are stored as offsets relative to
  // the start of the enclosing calldata head.
  Value relOffset = genLoad(ptr, sol::DataLocation::CallData, loc);
  unsigned neededLength = getCalldataEncodedTailSize(ty);
  assert(neededLength > 0 && "Expected non-zero calldata access size");

  Value callDataSize = b.create<yul::CallDataSizeOp>(loc);

  // The referenced head must fit entirely within the original calldata blob.
  // 'maxRelOffset' is the last relative offset whose required inline head is
  // still readable from 'baseAddr'.
  Value maxRelOffset = b.create<yul::SubOp>(
      loc, b.create<yul::SubOp>(loc, callDataSize, baseAddr),
      bExt.genI256Const(neededLength - 1));
  Value invalidOffset =
      bExt.genCmp(yul::CmpPredicate::sge, relOffset, maxRelOffset);
  genDebugRevertWithMsg(invalidOffset, msgOffset, loc);

  Value refAddr = b.create<yul::AddOp>(loc, baseAddr, relOffset);
  auto genLengthAndStrideGuards = [&](Value dataAddr, Value length,
                                      Value stride) {
    // Dynamic calldata payloads are materialized as {dataPtr, length}, so both
    // the decoded length and the implied payload range must stay in bounds.
    Value invalidLength =
        bExt.genCmp(yul::CmpPredicate::ugt, length,
                    bExt.genI256Const(APInt::getLowBitsSet(256, 64)));
    genDebugRevertWithMsg(invalidLength, msgLength, loc);

    Value maxDataAddr = b.create<yul::SubOp>(
        loc, callDataSize, b.create<yul::MulOp>(loc, length, stride));
    Value invalidStride =
        bExt.genCmp(yul::CmpPredicate::sgt, dataAddr, maxDataAddr);
    genDebugRevertWithMsg(invalidStride, msgStride, loc);
  };

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (!arrTy.isDynSized())
      // Fixed-size calldata arrays with dynamic children are still forwarded as
      // a single base address, as recursive element handling resolves child
      // heads.
      return refAddr;

    // Dynamic calldata arrays are passed as {dataPtr, size}.
    Value length = b.create<yul::CallDataLoadOp>(loc, refAddr);
    Value dataAddr = b.create<yul::AddOp>(loc, refAddr, bExt.genI256Const(32));
    genLengthAndStrideGuards(
        dataAddr, length,
        bExt.genI256Const(evm::getCallDataHeadSize(arrTy.getEltType())));
    return bExt.genLLVMStruct({dataAddr, length});
  }

  if (isa<sol::StringType>(ty)) {
    // Strings use the same {dataPtr, size} representation as bytes arrays.
    Value length = b.create<yul::CallDataLoadOp>(loc, refAddr);
    Value dataAddr = b.create<yul::AddOp>(loc, refAddr, bExt.genI256Const(32));
    genLengthAndStrideGuards(dataAddr, length, bExt.genI256Const(1));
    return bExt.genLLVMStruct({dataAddr, length});
  }

  if (isa<sol::StructType>(ty))
    // Nested calldata structs are forwarded by their head address.
    return refAddr;

  llvm_unreachable("Unexpected dynamically encoded calldata reference");
}

std::pair<Value, Value> evm::Builder::genPackedStorageAddrPair(
    Value baseSlot, Value idx, unsigned eltByteSize, bool isDataLeftAligned,
    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Consider that the slot may not be fully packed, e.g.,
  // 32 may not be evenly divisible by eltByteSize.
  unsigned itemsPerSlot = 32 / eltByteSize;
  Value itemsPerSlotV = bExt.genI256Const(itemsPerSlot);
  Value slotOffset = b.create<yul::ArithDivOp>(loc, idx, itemsPerSlotV);
  Value itemInSlot = b.create<yul::ArithModOp>(loc, idx, itemsPerSlotV);
  Value byteOffset = b.create<yul::MulOp>(loc, itemInSlot,
                                          bExt.genI256Const(eltByteSize, loc));
  if (isDataLeftAligned)
    byteOffset =
        b.create<yul::SubOp>(loc, bExt.genI256Const(31, loc), byteOffset);

  Value slot = b.create<yul::AddOp>(loc, baseSlot, slotOffset);
  return {slot, byteOffset};
}

Value evm::Builder::genPackedStorageAddr(Value baseSlot, Value idx, Type eltTy,
                                         bool isDataLeftAligned,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto [slot, byteOffset] = genPackedStorageAddrPair(
      baseSlot, idx, sol::getNumBytes(eltTy), isDataLeftAligned, loc);
  return bExt.genLLVMStruct({slot, byteOffset});
}

static Value genPunchHoleInValue(OpBuilder &b, Value value, Value shiftBits,
                                 unsigned numBits, Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);

  // holeMask = not(ones(numBits) << shiftBits)
  APInt ones = APInt::getLowBitsSet(256, numBits);
  Value shiftedOnes =
      b.create<yul::ArithShlOp>(loc, bExt.genI256Const(ones), shiftBits);
  Value holeMask = b.create<yul::XOrOp>(
      loc, shiftedOnes, bExt.genI256Const(APInt::getAllOnes(256)));

  // and(value, holeMask)
  return b.create<yul::AndOp>(loc, value, holeMask);
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

  if (sol::isBytesLikeType(eltTy)) {
    unsigned numBits = sol::getNumBytes(eltTy) * 8;
    if (numBits == 256)
      return value;
    return b.create<yul::ArithShlOp>(loc, value,
                                     bExt.genI256Const(256 - numBits, loc));
  }

  if (auto intTy = dyn_cast<IntegerType>(eltTy)) {
    // In packed storage, bool occupies one byte. Mask to the loaded storage
    // byte before applying generic bool cleanup.
    if (intTy.getWidth() == 1) {
      APInt mask = APInt::getLowBitsSet(256, 8);
      value = b.create<yul::AndOp>(loc, value, bExt.genI256Const(mask, loc));
    }
    return genIntCleanup(intTy.getWidth(), intTy.isSigned(), value, loc);
  }

  if (sol::isAddressLikeType(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 160);
    return b.create<yul::AndOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::EnumType>(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 8);
    return b.create<yul::AndOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::FuncRefType>(eltTy)) {
    APInt mask = APInt::getLowBitsSet(256, 64);
    return b.create<yul::AndOp>(loc, value, bExt.genI256Const(mask));
  }

  if (isa<sol::ExtFuncRefType>(eltTy))
    return b.create<yul::ArithShlOp>(loc, value, bExt.genI256Const(64, loc));

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
    addr = b.create<yul::AddOp>(loc, addr, bExt.genI256Const(32));
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
  Value length = b.create<yul::ArithShrOp>(loc, lengthSlot, one);
  Value isOutOfPlaceEnc = b.create<yul::AndOp>(loc, lengthSlot, one);
  Value isInPlace =
      bExt.genCmp(yul::CmpPredicate::eq, isOutOfPlaceEnc, bExt.genI256Const(0));

  Value maskedLength =
      b.create<yul::AndOp>(loc, length, bExt.genI256Const(0x7F));
  length = b.create<yul::ArithSelectOp>(loc, isInPlace, maskedLength, length)
               .getResult();

  Value lengthLT32 =
      bExt.genCmp(yul::CmpPredicate::ult, length, bExt.genI256Const(32));
  Value panicCond =
      bExt.genCmp(yul::CmpPredicate::eq, isOutOfPlaceEnc, lengthLT32);

  genPanic(mlir::evm::PanicCode::StorageEncodingError, panicCond);

  return length;
}

/// Calculates: and(val, ones(256) << (256 - (maskLen * 8)))
/// Precondition: maskLen > 0. Shifting an i256 by its own bitwidth (maskLen=0
/// -> shiftVal=256) produces poison under arith.shli semantics. All call sites
/// must guard against the empty-string case before invoking this helper.
static Value getI256MSBMaskedValue(OpBuilder &b, Value val, Value maskLen,
                                   Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);
  Value nbits = b.create<yul::MulOp>(loc, maskLen, bExt.genI256Const(8, loc));
  Value shiftVal =
      b.create<yul::SubOp>(loc, bExt.genI256Const(256, loc), nbits);
  Value mask = b.create<yul::ArithShlOp>(
      loc, bExt.genI256Const(APInt::getAllOnes(256), loc), shiftVal);
  return b.create<yul::AndOp>(loc, val, mask);
}

void evm::Builder::genClearStringStorageTail(Value dstAddr, Value oldLength,
                                             Value newLength,
                                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Only the out-of-place (long) form uses extra data slots; short strings
  // (≤31 bytes) are fully self-contained in the length slot.
  Value cleanCond =
      bExt.genCmp(yul::CmpPredicate::ugt, oldLength, bExt.genI256Const(31));
  auto ifClean = bExt.createIf(cleanCond);
  b.setInsertionPointToStart(&ifClean.getThenRegion().front());
  {
    Value dstDataArea =
        genDataAddrPtr(dstAddr, sol::DataLocation::Storage, loc);

    // If the new string fits in-place (newLength < 32), all old data slots
    // can be cleared. Otherwise, preserve the first ceil(newLength/32) slots.
    Value deleteStart = b.create<yul::AddOp>(
        loc, dstDataArea, bExt.genCeilDivision<32>(newLength));
    Value isInPlace =
        bExt.genCmp(yul::CmpPredicate::ult, newLength, bExt.genI256Const(32));
    deleteStart =
        b.create<yul::ArithSelectOp>(loc, isInPlace, dstDataArea, deleteStart);

    Value deleteEnd = b.create<yul::AddOp>(loc, dstDataArea,
                                           bExt.genCeilDivision<32>(oldLength));

    bExt.createCountedLoop(
        deleteStart, deleteEnd, bExt.genI256Const(1), ValueRange{},
        [&](OpBuilder &b, Location loc, Value indVar, ValueRange) {
          b.create<yul::SStoreOp>(loc, indVar, bExt.genI256Const(0, loc));
          return SmallVector<Value>{};
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
  Value outOfPlaceCond =
      bExt.genCmp(yul::CmpPredicate::ugt, length, bExt.genI256Const(31, loc));

  auto ifOutOfPlace = bExt.createIf(outOfPlaceCond, true);
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());

  Value dstDataArea = genDataAddrPtr(dstAddr, sol::DataLocation::Storage, loc);
  Value srcDataAddr = genDataAddrPtr(src, srcDataLoc, loc);
  Value loopEnd = b.create<yul::AndOp>(
      loc, length, bExt.genI256Const(~APInt(256, 0x1F), loc));

  // Copy the data in 32-byte chunks first. The loop variable is a byte offset.
  // For memory/calldata sources the byte offset indexes directly into the data
  // area. For storage sources, data is addressed by slot (32 bytes each), so
  // we divide the byte offset by 32 to obtain the slot index.
  bExt.createCountedLoop(
      bExt.genI256Const(0), loopEnd, bExt.genI256Const(32), ValueRange{},
      [&](OpBuilder &b, Location loc, Value i256IndVar, ValueRange) {
        Value slotIndVar = b.create<yul::ArithDivOp>(
            loc, i256IndVar, bExt.genI256Const(32, loc));
        Value srcIndVar =
            srcDataLoc == sol::DataLocation::Storage ? slotIndVar : i256IndVar;
        Value src = b.create<yul::AddOp>(loc, srcDataAddr, srcIndVar);
        Value val = genLoad(src, srcDataLoc, loc);
        Value dst = b.create<yul::AddOp>(loc, dstDataArea, slotIndVar);
        b.create<yul::SStoreOp>(loc, dst, val);
        return SmallVector<Value>{};
      });

  Value residualCond = bExt.genCmp(yul::CmpPredicate::ult, loopEnd, length);

  // Copy the remaining bytes (< 32) if the string length is not divisible
  // by 32.
  auto ifResidual = bExt.createIf(residualCond);
  {
    b.setInsertionPointToStart(&ifResidual.getThenRegion().front());
    Value residualLength =
        b.create<yul::AndOp>(loc, length, bExt.genI256Const(0x1F, loc));
    Value slotLoopEnd =
        b.create<yul::ArithDivOp>(loc, loopEnd, bExt.genI256Const(32, loc));
    Value srcResidualIdx =
        srcDataLoc == sol::DataLocation::Storage ? slotLoopEnd : loopEnd;
    Value lastVal =
        genLoad(b.create<yul::AddOp>(loc, srcDataAddr, srcResidualIdx),
                srcDataLoc, loc);
    Value maskedVal = getI256MSBMaskedValue(b, lastVal, residualLength, loc);
    Value dst = b.create<yul::AddOp>(loc, dstDataArea, slotLoopEnd);
    b.create<yul::SStoreOp>(loc, dst, maskedVal);
  }
  b.setInsertionPointAfter(ifResidual);

  // Store the string length.
  Value doubleLength =
      b.create<yul::MulOp>(loc, length, bExt.genI256Const(2, loc));
  b.create<yul::SStoreOp>(
      loc, dstAddr,
      b.create<yul::OrOp>(loc, doubleLength, bExt.genI256Const(1, loc)));

  // Handle in place case.
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value isNotEmptyCond =
        bExt.genCmp(yul::CmpPredicate::ne, length, bExt.genI256Const(0, loc));

    auto ifIsNotEmpty = bExt.createIf(isNotEmptyCond, true);
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
        b.create<yul::MulOp>(loc, length, bExt.genI256Const(2, loc));

    Value packedData = b.create<yul::OrOp>(loc, maskedVal, doubleLength);
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
    unsigned byteSize = sol::getNumBytes(eltTy);
    unsigned elemsPerSlot = 32 / byteSize;
    Value padded =
        b.create<yul::AddOp>(loc, len, bExt.genI256Const(elemsPerSlot - 1));
    return b.create<yul::ArithDivOp>(loc, padded,
                                     bExt.genI256Const(elemsPerSlot));
  }
  // Non-packable: each element occupies getStorageSlotCount(eltTy) slots.
  unsigned slotsPerElt = sol::getStorageSlotCount(eltTy);
  if (slotsPerElt == 1)
    return len;
  return b.create<yul::MulOp>(loc, len, bExt.genI256Const(slotsPerElt));
}

void evm::Builder::genClearStorageValue(Type ty, Value slot, Location loc) {
  mlir::solgen::BuilderExt bExt(b, loc);

  // Zero every storage slot that \p ty occupies at \p slot.
  auto genZeroStorageSlots = [&](Type ty, Value slot) {
    unsigned slotCount = sol::getStorageSlotCount(ty);
    for (unsigned i = 0; i < slotCount; ++i) {
      Value slotAddr = b.create<yul::AddOp>(loc, slot, bExt.genI256Const(i));
      b.create<yul::SStoreOp>(loc, slotAddr, bExt.genI256Const(0, loc));
    }
  };

  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    if (arrTy.isDynSized()) {
      Value oldLen = genLoad(slot, sol::DataLocation::Storage, loc);
      genStore(bExt.genI256Const(0, loc), slot, sol::DataLocation::Storage, loc);
      genClearStorageArrayTail(slot, arrTy, bExt.genI256Const(0, loc), oldLen,
                               /*isDecrement=*/false, loc);
    } else if (sol::hasDynamicallySizedElt(arrTy.getEltType())) {
      // Fixed-size array with complex element type (dyn array, string,
      // struct): iterate each element and recurse.
      Type eltTy = arrTy.getEltType();
      unsigned size = arrTy.getSize();
      unsigned slotsPerElt = sol::getStorageSlotCount(eltTy);
      for (unsigned i = 0; i < size; ++i) {
        Value eltSlot =
            b.create<yul::AddOp>(loc, slot, bExt.genI256Const(i * slotsPerElt));
        genClearStorageValue(eltTy, eltSlot, loc);
      }
    } else {
      // Scalar/packed fixed-size array (e.g. uint72[2]).
      genZeroStorageSlots(ty, slot);
    }
    return;
  }

  if (isa<sol::StringType>(ty)) {
    Value zero = bExt.genI256Const(0, loc);
    Value strOldLen = genStorageStringLength(
        genLoad(slot, sol::DataLocation::Storage, loc), loc);
    genClearStringStorageTail(slot, strOldLen, zero, loc);
    b.create<yul::SStoreOp>(loc, slot, zero);
    return;
  }

  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    // Packed members can share a slot; dedup their zeroing. Members with
    // dynamically-sized elements always recurse, regardless of slot sharing.
    llvm::SmallSet<uint64_t, 8> clearedSlots;
    for (unsigned m = 0; m < structTy.getMemberTypes().size(); ++m) {
      Type memberTy = structTy.getMemberTypes()[m];
      auto [slotOff, byteOff] = structTy.getStorageMemberOffset(m);
      (void)byteOff;
      if (!sol::hasDynamicallySizedElt(memberTy) &&
          !clearedSlots.insert(slotOff).second)
        continue;

      Value memberSlot =
          b.create<yul::AddOp>(loc, slot, bExt.genI256Const(slotOff));
      genClearStorageValue(memberTy, memberSlot, loc);
    }
    return;
  }

  // Scalar or packed value.
  genZeroStorageSlots(ty, slot);
}

// Clears storage slots for elements [startIdx, endIdx) in a storage array.
// Used when a dynamic array shrinks or a shorter source is copied into a
// longer destination.
void evm::Builder::genClearStorageArrayTail(Value arraySlot,
                                            sol::ArrayType arrTy,
                                            Value startIdx, Value endIdx,
                                            bool isDecrement,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Type eltTy = arrTy.getEltType();
  // Multiple elements share one slot when byteSize <= 16 (elemsPerSlot >= 2).
  bool trulyPacked =
      sol::canBePacked(eltTy) && sol::getNumBytes(eltTy) <= 16;

  yul::IfOp ifClear = nullptr;
  if (!isDecrement) {
    Value needsClear = bExt.genCmp(yul::CmpPredicate::ult, startIdx, endIdx);
    ifClear = bExt.createIf(needsClear, /*withElseRegion=*/false);
    b.setInsertionPointToStart(&ifClear.getThenRegion().front());
  }
  {
    // For dynamic arrays the data lives at keccak256(arraySlot). For static
    // arrays it lives directly at arraySlot. Computed here (inside the guard)
    // so the hash is only emitted when the range is known non-empty.
    Value dataStart =
        arrTy.isDynSized()
            ? genDataAddrPtr(arraySlot, sol::DataLocation::Storage, loc)
            : arraySlot;
    Value deleteStart = b.create<yul::AddOp>(
        loc, dataStart, genStorageArraySlotCount(startIdx, eltTy, loc));

    // Packed-type partial clear: when startIdx falls in the middle of a slot
    // (i.e. startIdx % elemsPerSlot != 0), the slot at (deleteStart−1) is only
    // partially occupied by still-valid elements. The upper bytes belonging to
    // the removed elements must be zeroed while the lower bytes are preserved.
    if (trulyPacked) {
      unsigned byteSize = sol::getNumBytes(eltTy);
      unsigned elemsPerSlot = 32 / byteSize;
      Value remainder = b.create<yul::ArithModOp>(
          loc, startIdx, bExt.genI256Const(elemsPerSlot));
      Value hasRemainder =
          bExt.genCmp(yul::CmpPredicate::ne, remainder, bExt.genI256Const(0));
      // For isDecrement, the else branch handles the case where the popped
      // element is the sole occupant of its slot and the whole slot must be
      // zeroed. No loop is needed: a pop removes exactly one element, so at
      // most one full slot can become empty.
      auto ifPartial = bExt.createIf(hasRemainder,
                                     /*withElseRegion=*/isDecrement, loc);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&ifPartial.getThenRegion().front());
        // bitsToKeep = remainder * byteSize * 8,
        // mask = allOnes >> (256 - bitsToKeep)
        Value bitsToKeep = b.create<yul::MulOp>(
            loc,
            b.create<yul::MulOp>(loc, remainder, bExt.genI256Const(byteSize)),
            bExt.genI256Const(8));
        Value allOnes = bExt.genI256Const(APInt::getAllOnes(256), loc);
        Value shiftAmt =
            b.create<yul::SubOp>(loc, bExt.genI256Const(256), bitsToKeep);
        Value mask = b.create<yul::ArithShrOp>(loc, allOnes, shiftAmt);
        Value partialSlot =
            b.create<yul::SubOp>(loc, deleteStart, bExt.genI256Const(1));
        Value slotVal = genLoad(partialSlot, sol::DataLocation::Storage, loc);
        genStore(b.create<yul::AndOp>(loc, slotVal, mask), partialSlot,
                 sol::DataLocation::Storage, loc);
      }
      if (isDecrement) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(&ifPartial.getElseRegion().front());
        genClearStorageValue(eltTy, deleteStart, loc);
      }
    }

    if (isDecrement) {
      if (!trulyPacked) {
        // Pop of exactly one element: genClearStorageValue handles all slots of
        // the element (including struct members, nested arrays, etc.) from the
        // base slot, so a single call is sufficient.
        genClearStorageValue(eltTy, deleteStart, loc);
      }
      // trulyPacked: fully handled by the ifPartial block above.
    } else {
      Value deleteEnd = b.create<yul::AddOp>(
          loc, dataStart, genStorageArraySlotCount(endIdx, eltTy, loc));
      Value numSlots = b.create<yul::SubOp>(loc, deleteEnd, deleteStart);
      // For non-packed types getStorageSlotCount(eltTy) >= 1 and slotStep
      // equals the element's full slot footprint, so each iteration clears one
      // complete element.
      // For truly-packed types (multiple elements per slot) slotStep == 1 and
      // the loop iterates slot-by-slot; genClearStorageValue zeroes the whole
      // slot. The partial boundary slot at (deleteStart - 1) was already
      // handled by the ifPartial block above.
      unsigned slotStep = sol::getStorageSlotCount(eltTy);
      bExt.createCountedLoop(
          /*lowerBound=*/bExt.genI256Const(0),
          /*upperBound=*/numSlots,
          /*step=*/bExt.genI256Const(slotStep),
          /*initArgs=*/ValueRange{},
          [&](OpBuilder &b, Location loc, Value i, ValueRange) {
            Value slot = b.create<yul::AddOp>(loc, deleteStart, i);
            genClearStorageValue(eltTy, slot, loc);
            return SmallVector<Value>{};
          },
          /*fullUnroll*/ false, loc);
    }
  }
  if (ifClear)
    b.setInsertionPointAfter(ifClear);
}

void evm::Builder::genResizeDynStorageArray(Value arraySlot, Value newLen,
                                            Type eltTy,
                                            std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  // Panic if newLen > type(uint64).max (ResourceError / too-large array).
  mlir::solgen::BuilderExt bExt(b, loc);
  Value panicCond =
      bExt.genCmp(yul::CmpPredicate::ugt, newLen,
                  bExt.genI256Const(APInt::getLowBitsSet(256, 64), loc));
  genPanic(PanicCode::ResourceError, panicCond, loc);

  // Read the old length before overwriting it.
  Value oldLen = genLoad(arraySlot, sol::DataLocation::Storage, loc);

  // Write the new length.
  genStore(newLen, arraySlot, sol::DataLocation::Storage, loc);

  // Zero out storage slots that fall outside the new range.
  auto dynArrTy = sol::ArrayType::get(b.getContext(), /*size=*/-1, eltTy,
                                      sol::DataLocation::Storage);
  genClearStorageArrayTail(arraySlot, dynArrTy, newLen, oldLen,
                           /*isDecrement=*/false, loc);
}

Value evm::Builder::genCopyStringDataToMemory(Value src, Type ty,
                                              Value dstDataAddr,
                                              std::optional<Location> locArg,
                                              bool withCleanup) {
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

  if (withCleanup) {
    // Canonicalize trailing bytes after variable-length payload copies by
    // clearing the word at dst + length.
    mlir::solgen::BuilderExt bExt(b, loc);
    Value cleanupAddr = b.create<yul::AddOp>(loc, dstDataAddr, length);
    b.create<yul::MStoreOp>(loc, cleanupAddr, bExt.genI256Const(0, loc));
  }

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
  Value isOutOfPlaceEnc = b.create<yul::AndOp>(loc, lengthSlot, one);
  Value isInPlace = bExt.genCmp(yul::CmpPredicate::eq, isOutOfPlaceEnc, zero);

  auto ifInPlace = bExt.createIf(isInPlace, true);
  // In-place path: the data bytes live in the high 31 bytes of the length slot
  // itself. Strip the low byte (the length encoding) and write one word.
  b.setInsertionPointToStart(&ifInPlace.getThenRegion().front());
  {
    Value val = b.create<yul::AndOp>(loc, lengthSlot,
                                     bExt.genI256Const(~APInt(256, 0xFF), loc));
    b.create<yul::MStoreOp>(loc, dstDataAddr, val);
  }
  // Out-of-place path: copy 32-byte storage slots to memory word by word.
  b.setInsertionPointToStart(&ifInPlace.getElseRegion().front());
  {
    auto srcDataAddr = genDataAddrPtr(src, sol::DataLocation::Storage, loc);
    bExt.createCountedLoop(
        bExt.genI256Const(0), length, bExt.genI256Const(32), ValueRange{},
        [&](OpBuilder &b, Location loc, Value i256IndVar, ValueRange) {
          Value slotOffset = b.create<yul::ArithDivOp>(
              loc, i256IndVar, bExt.genI256Const(32, loc));
          Value src = b.create<yul::AddOp>(loc, srcDataAddr, slotOffset);
          Value val = b.create<yul::SLoadOp>(loc, src);
          Value dst = b.create<yul::AddOp>(loc, dstDataAddr, i256IndVar);
          b.create<yul::MStoreOp>(loc, dst, val);
          return SmallVector<Value>{};
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
      b.create<yul::MulOp>(loc, offset, bExt.genI256Const(8, loc));
  Value punchedSlot = genPunchHoleInValue(b, slot, bitOffset, numBits, loc);
  Value shiftedByte = b.create<yul::ArithShlOp>(loc, intVal, bitOffset);
  return b.create<yul::OrOp>(loc, punchedSlot, shiftedByte);
}

void evm::Builder::genPushToString(Value srcAddr, Value value,
                                   std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value data = genLoad(srcAddr, sol::DataLocation::Storage, loc);
  Value oldLength = genStorageStringLength(data, loc);

  Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, oldLength,
                                bExt.genI256Const(APInt(256, 1).shl(64), loc));
  genPanic(mlir::evm::PanicCode::ResourceError, panicCond);

  Value isOutOfPlace = bExt.genCmp(yul::CmpPredicate::ugt, oldLength,
                                   bExt.genI256Const(31, loc));

  auto ifOutOfPlace = bExt.createIf(isOutOfPlace, true);
  // Out of place path
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());
  {
    Value newLength =
        b.create<yul::AddOp>(loc, data, bExt.genI256Const(2, loc));
    // Update the length.
    b.create<yul::SStoreOp>(loc, srcAddr, newLength);
    Value dataPtr = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
    auto [slotNum, offset] =
        genPackedStorageAddrPair(dataPtr, oldLength, /*eltByteSize*/ 1,
                                 /*isDataLeftAligned*/ true, loc);
    Value slotVal = genLoad(slotNum, sol::DataLocation::Storage, loc);
    Value updatedSlot =
        genInsertIntToSlot(slotVal, offset, value, /*numBits*/ 8, loc);
    b.create<yul::SStoreOp>(loc, slotNum, updatedSlot);
  }

  // In place path
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value convertToUnpacked = bExt.genCmp(yul::CmpPredicate::eq, oldLength,
                                          bExt.genI256Const(31, loc));

    auto ifConvertToUnpacked = bExt.createIf(convertToUnpacked, true);
    b.setInsertionPointToStart(&ifConvertToUnpacked.getThenRegion().front());
    {
      // Here we have special case when array switches from short array
      // to long array. We need to copy data.
      Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);

      Value mask = bExt.genI256Const(~APInt(256, 0xFF), loc);
      Value maskedData = b.create<yul::AndOp>(loc, data, mask);
      Value res = b.create<yul::OrOp>(loc, value, maskedData);
      b.create<yul::SStoreOp>(loc, dataSlot, res);
      // New length is 32, encoded as (32 * 2 + 1)
      b.create<yul::SStoreOp>(loc, srcAddr, bExt.genI256Const(65, loc));
    }

    b.setInsertionPointToStart(&ifConvertToUnpacked.getElseRegion().front());
    {
      Value offset =
          b.create<yul::SubOp>(loc, bExt.genI256Const(31, loc), oldLength);
      Value updatedSlot =
          genInsertIntToSlot(data, offset, value, /*numBits*/ 8, loc);
      Value res =
          b.create<yul::AddOp>(loc, updatedSlot, bExt.genI256Const(2, loc));
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

  Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, oldLength,
                                bExt.genI256Const(APInt(256, 1).shl(64), loc));
  genPanic(mlir::evm::PanicCode::ResourceError, panicCond);

  Type i256Ty = IntegerType::get(b.getContext(), 256,
                                 IntegerType::SignednessSemantics::Signless);
  Type resTy =
      LLVM::LLVMStructType::getLiteral(b.getContext(), {i256Ty, i256Ty});
  Type bytes1Ty = mlir::sol::FixedBytesType::get(b.getContext(), /*size*/ 1);

  Value isOutOfPlace = bExt.genCmp(yul::CmpPredicate::ugt, oldLength,
                                   bExt.genI256Const(31, loc));

  auto ifOutOfPlace = bExt.createIf(resTy, isOutOfPlace);
  // Out of place path
  b.setInsertionPointToStart(&ifOutOfPlace.getThenRegion().front());
  {
    Value newLength =
        b.create<yul::AddOp>(loc, data, bExt.genI256Const(2, loc));
    Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage);
    // Update the length.
    b.create<yul::SStoreOp>(loc, srcAddr, newLength);
    b.create<yul::YieldOp>(loc,
                           genPackedStorageAddr(dataSlot, oldLength, bytes1Ty,
                                                /*isDataLeftAligned*/ true));
  }

  // In place path
  b.setInsertionPointToStart(&ifOutOfPlace.getElseRegion().front());
  {
    Value convertToUnpacked = bExt.genCmp(yul::CmpPredicate::eq, oldLength,
                                          bExt.genI256Const(31, loc));

    auto ifConvertToUnpacked = bExt.createIf(resTy, convertToUnpacked);
    b.setInsertionPointToStart(&ifConvertToUnpacked.getThenRegion().front());
    {
      // Here we have special case when array switches from short array
      // to long array. We need to copy data.
      Value dataSlot = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
      Value mask = bExt.genI256Const(~APInt(256, 0xFF), loc);
      Value maskedOldData = b.create<yul::AndOp>(loc, data, mask);
      b.create<yul::SStoreOp>(loc, dataSlot, maskedOldData);
      // New length is 32, encoded as (32 * 2 + 1)
      b.create<yul::SStoreOp>(loc, srcAddr, bExt.genI256Const(65, loc));
      b.create<yul::YieldOp>(loc,
                             genPackedStorageAddr(dataSlot, oldLength, bytes1Ty,
                                                  /*isDataLeftAligned*/ true));
    }

    b.setInsertionPointToStart(&ifConvertToUnpacked.getElseRegion().front());
    {
      Value res = b.create<yul::AddOp>(loc, data, bExt.genI256Const(2, loc));
      b.create<yul::SStoreOp>(loc, srcAddr, res);
      b.create<yul::YieldOp>(loc,
                             genPackedStorageAddr(srcAddr, oldLength, bytes1Ty,
                                                  /*isDataLeftAligned*/ true));
    }
    b.setInsertionPointAfter(ifConvertToUnpacked);
    b.create<yul::YieldOp>(loc, ifConvertToUnpacked.getResult(0));
  }
  b.setInsertionPointAfter(ifOutOfPlace);

  return ifOutOfPlace.getResult(0);
}

Value evm::Builder::genStringItemAddress(Value srcAddr, Value idx,
                                         std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Type bytes1Ty = mlir::sol::FixedBytesType::get(b.getContext(), /*size*/ 1);
  Value data = genLoad(srcAddr, sol::DataLocation::Storage, loc);
  Value length = genStorageStringLength(data, loc);

  auto panicCond = bExt.genCmp(yul::CmpPredicate::uge, idx, length);
  genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

  Value isOutOfPlace =
      bExt.genCmp(yul::CmpPredicate::ugt, length, bExt.genI256Const(31, loc));

  Value dataSlot =
      b.create<yul::ArithSelectOp>(
           loc, isOutOfPlace,
           genDataAddrPtr(srcAddr, sol::DataLocation::Storage), srcAddr)
          .getResult();

  return genPackedStorageAddr(dataSlot, idx, bytes1Ty,
                              /*isDataLeftAligned*/ true);
}

void evm::Builder::genPopString(Value srcAddr, Value oldData, Value length,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value convertToPacked =
      bExt.genCmp(yul::CmpPredicate::eq, length, bExt.genI256Const(32, loc));

  auto ifConvertToPacked = bExt.createIf(convertToPacked, true);
  b.setInsertionPointToStart(&ifConvertToPacked.getThenRegion().front());
  {
    // Special case: array transitions from out-of-place (length == 32) to
    // in-place (length == 31). Copy the 31 MSB remaining bytes from the data
    // slot back into srcAddr in packed encoding.
    // The new length encoding is 31*2 = 62.
    Value dataPos = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
    Value slotData = genLoad(dataPos, sol::DataLocation::Storage, loc);
    Value maskedData = b.create<yul::AndOp>(
        loc, slotData, bExt.genI256Const(~APInt(256, 0xFF), loc));
    b.create<yul::SStoreOp>(
        loc, srcAddr,
        b.create<yul::OrOp>(loc, maskedData, bExt.genI256Const(62, loc)));
    b.create<yul::SStoreOp>(loc, dataPos, bExt.genI256Const(0, loc));
  }

  b.setInsertionPointToStart(&ifConvertToPacked.getElseRegion().front());
  {
    Value newLen = b.create<yul::SubOp>(loc, length, bExt.genI256Const(1));
    Value isPacked =
        bExt.genCmp(yul::CmpPredicate::ult, length, bExt.genI256Const(32, loc));

    auto ifPacked = bExt.createIf(isPacked, true);
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
          b.create<yul::MulOp>(loc, newLen, bExt.genI256Const(8, loc));
      Value mask = b.create<yul::XOrOp>(
          loc, b.create<yul::ArithShrOp>(loc, allOnes, nbits), allOnes);
      Value maskedData = b.create<yul::AndOp>(loc, oldData, mask);
      Value dLen = b.create<yul::MulOp>(loc, newLen, bExt.genI256Const(2, loc));
      b.create<yul::SStoreOp>(loc, srcAddr,
                              b.create<yul::OrOp>(loc, maskedData, dLen));
    }

    b.setInsertionPointToStart(&ifPacked.getElseRegion().front());
    {
      Value dataPtr = genDataAddrPtr(srcAddr, sol::DataLocation::Storage, loc);
      auto [slotNum, offset] =
          genPackedStorageAddrPair(dataPtr, newLen, /*eltByteSize*/ 1,
                                   /*isDataLeftAligned*/ true, loc);
      Value slot = genLoad(slotNum, sol::DataLocation::Storage, loc);
      Value updatedSlot =
          genInsertIntToSlot(slot, offset, bExt.genI256Const(0, loc),
                             /*numBits*/ 8, loc);
      b.create<yul::SStoreOp>(loc, slotNum, updatedSlot);
      Value newData = b.create<yul::SubOp>(loc, oldData, bExt.genI256Const(2));
      b.create<yul::SStoreOp>(loc, srcAddr, newData);
    }
    b.setInsertionPointAfter(ifPacked);
  }
  b.setInsertionPointAfter(ifConvertToPacked);
}

// Copies from (srcAddr, srcDataLoc) to (dstAddr, dstDataLoc).
// Dispatch is on (dstTy, srcDataLoc, dstDataLoc, element packing).
// +----------------------------+----------------------------------------------+
// | Condition                  | Action                                       |
// +----------------------------+----------------------------------------------+
// | Scalar: int, addr,         | Plain load/store. Encoding is identical      |
// |   enum, FuncRef            | across Stg, Mem, and CD.                     |
// +----------------------------+----------------------------------------------+
// | bytesN, ExtFuncRef         | Mem/CD->Mem/CD: plain 32-byte load/store.    |
// |                            | Mem/CD->Stg: load, SHR alignShift to         |
// |                            |   right-align, then store.                   |
// |                            | Stg->Mem/CD: load, SHL alignShift to         |
// |                            |   left-align, then store.                    |
// |                            | Stg->Stg: plain 32-byte load/store.          |
// |                            | (alignShift = (32-N)*8 for bytesN; 64        |
// |                            |   for ExtFuncRef)                            |
// +----------------------------+----------------------------------------------+
// | StringType (string/bytes)  | any->Stg: genCopyStringToStorage; handles    |
// |                            |   inline <-> out-of-place transitions.       |
// |                            | any->Mem: genCopyStringDataToMemory +        |
// |                            |   store length.                              |
// +----------------------------+----------------------------------------------+
// | StructType                 | non-Stg dst: llvm_unreachable; must be       |
// |                            |   lowered by DataLocCastOpLowering.          |
// |                            | any->Stg: per-member via                     |
// |                            |   StructEncodeMemberReader +                 |
// |                            |   StructMemberWriter.                        |
// +----------------------------+----------------------------------------------+
// | Array, packed dst elt      | Stg->Stg, same density: raw slot copy.       |
// |                            | Stg->Stg, packed src: per-element extract-   |
// |                            |   convert-repack loop.                       |
// |                            | Stg->Stg, npacked src: compact-write loop,   |
// |                            |   one sload per source element.              |
// |                            | Mem/CD->Stg: compact-write loop: load +      |
// |                            |   right-align bytesN + pack into slots.      |
// +----------------------------+----------------------------------------------+
// | Array, packed src elt ->   | Stg->Stg: compact-read loop: extract +       |
// |   npacked dst              |   optional bytes widen-shift + sstore        |
// |                            |   per element.                               |
// |                            | Stg->Mem/CD: compact-read loop: extract +    |
// |                            |   cleanup + store (32-byte word per elt).    |
// +----------------------------+----------------------------------------------+
// | Array, generic fallback    | any->any: yul.for over [0, len):             |
// |   (npacked on both sides,  |   genAddrAtIdx, then resolve CD/Mem ref      |
// |   or neither side is Stg)  |   pointers, then genCopy per element.        |
// +----------------------------+----------------------------------------------+
void evm::Builder::genCopy(Type srcTy, Type dstTy, Value srcAddr, Value dstAddr,
                           sol::DataLocation srcDataLoc,
                           sol::DataLocation dstDataLoc,
                           std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  assert(!isa<sol::MappingType>(srcTy) && !isa<sol::MappingType>(dstTy));

  // Compile-time SSA identity check.
  if (srcAddr == dstAddr)
    return;

  bool srcIsStorage = srcDataLoc == sol::DataLocation::Storage;
  bool dstIsStorage = dstDataLoc == sol::DataLocation::Storage;

  // Runtime self-assignment guard for storage aggregate copies: two distinct
  // SSA values may still resolve to the same storage slot at runtime.
  std::optional<OpBuilder::InsertionGuard> selfAssignGuard;
  if (srcIsStorage && dstIsStorage && sol::isNonPtrRefType(dstTy)) {
    Value notSelf = bExt.genCmp(yul::CmpPredicate::ne, srcAddr, dstAddr, loc);
    auto ifOp = bExt.createIf(notSelf, /*withElseRegion=*/false, loc);
    selfAssignGuard.emplace(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  }

  // Handle the destination array length change.
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
      if (dstIsStorage)
        genResizeDynStorageArray(dstAddr, length, dstArrTy.getEltType(), loc);
      else
        genStore(length, dstAddr, dstDataLoc, loc);

      dstDataAddr = genDataAddrPtr(dstAddr, dstDataLoc, loc);
    } else {
      // Static destination array: loop over the source length, then zero any
      // tail slots in the destination that the source doesn't cover.
      length = bExt.genI256Const(srcArrTy.getSize());
      dstDataAddr = dstAddr;
      srcDataAddr = srcAddr;
      if (dstIsStorage && srcArrTy.getSize() < dstArrTy.getSize()) {
        genClearStorageArrayTail(dstAddr, dstArrTy,
                                 bExt.genI256Const(srcArrTy.getSize()),
                                 bExt.genI256Const(dstArrTy.getSize()),
                                 /*isDecrement=*/false, loc);
      }
    }

    Type srcEltTy = srcArrTy.getEltType();
    Type dstEltTy = dstArrTy.getEltType();

    unsigned srcEltByteSize =
        sol::canBePacked(srcEltTy) ? sol::getNumBytes(srcEltTy) : 0;
    unsigned dstEltByteSize =
        sol::canBePacked(dstEltTy) ? sol::getNumBytes(dstEltTy) : 0;
    bool srcPacked =
        sol::canBePacked(srcEltTy) && sol::getNumBytes(srcEltTy) <= 16;
    bool dstPacked =
        sol::canBePacked(dstEltTy) && sol::getNumBytes(dstEltTy) <= 16;

    // Packed element types require specialised loops when storage is involved.

    // Storage -> Storage
    if (srcIsStorage && dstIsStorage) {
      // Identical layout on both sides: raw slot copy.
      if (srcPacked && dstPacked &&
          (sol::getNumBytes(srcEltTy) == sol::getNumBytes(dstEltTy))) {
        unsigned itemsPerSlot = 32 / dstEltByteSize;
        unsigned eltBits = dstEltByteSize * 8;
        APInt fullSlotMask = APInt::getLowBitsSet(256, itemsPerSlot * eltBits);

        Value numFullSlots = b.create<yul::ArithDivOp>(
            loc, length, bExt.genI256Const(itemsPerSlot));
        Value remainder = b.create<yul::ArithModOp>(
            loc, length, bExt.genI256Const(itemsPerSlot));

        bExt.createCountedLoop(
            bExt.genI256Const(0), numFullSlots, bExt.genI256Const(1),
            ValueRange{}, [&](OpBuilder &b, Location loc, Value i, ValueRange) {
              Value srcSlot = b.create<yul::AddOp>(loc, srcDataAddr, i);
              Value dstSlot = b.create<yul::AddOp>(loc, dstDataAddr, i);
              Value raw = b.create<yul::SLoadOp>(loc, srcSlot);
              Value masked = b.create<yul::AndOp>(
                  loc, raw, bExt.genI256Const(fullSlotMask));
              b.create<yul::SStoreOp>(loc, dstSlot, masked);
              return SmallVector<Value>{};
            });

        // Handle the final partial slot (if length % itemsPerSlot != 0).
        Value hasRemainder =
            bExt.genCmp(yul::CmpPredicate::ne, remainder, bExt.genI256Const(0));
        auto remIf = bExt.createIf(hasRemainder);
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&remIf.getThenRegion().front());

          Value lastSrcSlot =
              b.create<yul::AddOp>(loc, srcDataAddr, numFullSlots);
          Value lastDstSlot =
              b.create<yul::AddOp>(loc, dstDataAddr, numFullSlots);
          Value lastRaw = b.create<yul::SLoadOp>(loc, lastSrcSlot);
          Value remBits =
              b.create<yul::MulOp>(loc, remainder, bExt.genI256Const(eltBits));
          Value shifted =
              b.create<yul::ArithShlOp>(loc, bExt.genI256Const(1), remBits);
          Value partialMask =
              b.create<yul::SubOp>(loc, shifted, bExt.genI256Const(1));
          Value lastMasked = b.create<yul::AndOp>(loc, lastRaw, partialMask);
          b.create<yul::SStoreOp>(loc, lastDstSlot, lastMasked);
        }
        return;
      }

      // Packed src -> packed dst, different element sizes
      if (srcPacked && dstPacked) {
        emitRepackStorageToStorageCopyLoop(b, loc, length, srcDataAddr,
                                           dstDataAddr, srcEltTy,
                                           srcEltByteSize, dstEltByteSize);
        return;
      }

      // Non-packed src -> packed dst.
      if (dstPacked) {
        assert(srcEltByteSize > dstEltByteSize);
        unsigned narrowShift = isa<sol::FixedBytesType>(srcEltTy)
                                   ? (srcEltByteSize - dstEltByteSize) * 8
                                   : 0;
        APInt dstMask = APInt::getLowBitsSet(256, dstEltByteSize * 8);
        emitCompactStorageArrayWriteLoop(
            b, loc, length, dstDataAddr, srcDataAddr, dstEltByteSize,
            dstArrTy.isDynSized(),
            [&](OpBuilder &b, Location loc, Value srcAddr, Value accum,
                Value shiftBits) {
              Value loaded = b.create<yul::SLoadOp>(loc, srcAddr);
              Value narrowed = nullptr;
              if (narrowShift > 0)
                narrowed = b.create<yul::ArithShrOp>(
                    loc, loaded, bExt.genI256Const(narrowShift));
              else
                narrowed = loaded;
              Value masked = b.create<yul::AndOp>(loc, narrowed,
                                                  bExt.genI256Const(dstMask));
              Value shifted = b.create<yul::ArithShlOp>(loc, masked, shiftBits);
              Value newAccum = b.create<yul::OrOp>(loc, accum, shifted);
              // One storage slot per src element.
              Value nextSrcAddr =
                  b.create<yul::AddOp>(loc, srcAddr, bExt.genI256Const(1));
              return std::make_pair(newAccum, nextSrcAddr);
            });
        return;
      }

      // Packed src -> non-packed dst.
      if (srcPacked) {
        assert(dstEltByteSize > srcEltByteSize);
        unsigned widenShift = isa<sol::FixedBytesType>(dstEltTy)
                                  ? (dstEltByteSize - srcEltByteSize) * 8
                                  : 0;
        emitCompactStorageArrayReadLoop(
            b, loc, length, dstDataAddr, srcDataAddr,
            /*dstStride=*/bExt.genI256Const(1), srcEltByteSize,
            srcArrTy.isDynSized(),
            [&](OpBuilder &b, Location loc, Value slotVal, Value shiftBits,
                Value dstAddr) {
              Value extracted =
                  b.create<yul::ArithShrOp>(loc, slotVal, shiftBits);
              Value widened = b.create<yul::ArithShlOp>(
                  loc, extracted, bExt.genI256Const(widenShift));
              APInt maskBits = APInt::getBitsSet(
                  256, widenShift, widenShift + srcEltByteSize * 8);
              Value val = b.create<yul::AndOp>(loc, widened,
                                               bExt.genI256Const(maskBits));
              genStore(val, dstAddr, sol::DataLocation::Storage, loc);
            });
        return;
      }
    }

    // Storage src packed -> Memory: extract and widen each element.
    if (srcPacked && srcIsStorage) {
      emitCompactStorageArrayReadLoop(
          b, loc, length, dstDataAddr, srcDataAddr, bExt.genI256Const(32),
          srcEltByteSize, srcArrTy.isDynSized(),
          [&](OpBuilder &b, Location loc, Value slotVal, Value shiftBits,
              Value dstAddr) {
            Value shifted = b.create<yul::ArithShrOp>(loc, slotVal, shiftBits);
            Value val = genCleanupPackedStorageValue(srcEltTy, shifted, loc);
            genStore(val, dstAddr, dstDataLoc, loc);
          });
      return;
    }

    // Memory/CallData -> storage dst packed: pack source elements into slots.
    if (dstPacked && dstIsStorage) {
      unsigned numBits = dstEltByteSize * 8;
      APInt mask = APInt::getLowBitsSet(256, numBits);
      emitCompactStorageArrayWriteLoop(
          b, loc, length, dstDataAddr, srcDataAddr, dstEltByteSize,
          dstArrTy.isDynSized(),
          [&](OpBuilder &b, Location loc, Value srcAddr, Value accum,
              Value shiftBits) {
            Value val = genLoad(srcAddr, srcDataLoc, loc);
            // Validate-and-revert each element from calldata
            // before packing (e.g. signextend check for signed int types).
            if (srcDataLoc == sol::DataLocation::CallData)
              val = genCleanup(dstEltTy, val, loc, /*shouldRevert=*/true);

            // ExtFuncRef is not handled here, as it occupies a full slot in
            // array.
            if (isa<sol::FixedBytesType>(dstEltTy))
              val = b.create<yul::ArithShrOp>(loc, val,
                                              bExt.genI256Const(256 - numBits));

            Value masked =
                b.create<yul::AndOp>(loc, val, bExt.genI256Const(mask));
            Value shifted = b.create<yul::ArithShlOp>(loc, masked, shiftBits);
            Value newAccum = b.create<yul::OrOp>(loc, accum, shifted);
            Value nextSrcAddr =
                b.create<yul::AddOp>(loc, srcAddr, bExt.genI256Const(32));
            return std::make_pair(newAccum, nextSrcAddr);
          });
      return;
    }

    // Resolve a element address before recursing:
    // - calldata with dynamic children: load the relative offset and validate
    //   bounds via genCalldataAccessRef.
    // - memory reference types: dereference the pointer stored in the slot.
    // - storage: addresses are direct.
    auto resolveEltAddr = [&](Value addr, Type eltTy,
                              sol::DataLocation dataLoc) -> Value {
      if (dataLoc == sol::DataLocation::CallData &&
          sol::hasDynamicallySizedElt(eltTy))
        return genCalldataAccessRef(eltTy, srcDataAddr, addr,
                                    /*isNonABI=*/true, loc);
      if (dataLoc == sol::DataLocation::Memory && sol::isNonPtrRefType(eltTy))
        return genLoad(addr, dataLoc, loc);
      return addr;
    };

    bExt.createCountedLoop(
        bExt.genI256Const(0), length, bExt.genI256Const(1), ValueRange{},
        [&](OpBuilder &b, Location loc, Value i256IndVar, ValueRange) {
          Value srcAddrI =
              genAddrAtIdx(srcDataAddr, i256IndVar, srcArrTy, srcDataLoc, loc);
          Value dstAddrI =
              genAddrAtIdx(dstDataAddr, i256IndVar, dstArrTy, dstDataLoc, loc);

          srcAddrI = resolveEltAddr(srcAddrI, srcEltTy, srcDataLoc);
          dstAddrI = resolveEltAddr(dstAddrI, dstEltTy, dstDataLoc);

          genCopy(srcEltTy, dstEltTy, srcAddrI, dstAddrI, srcDataLoc,
                  dstDataLoc, loc);
          return SmallVector<Value>{};
        });
  } else if (sol::canBePacked(dstTy)) {
    // This path is only taken when copying unpacked array elements.
    assert((!srcIsStorage && !dstIsStorage) || sol::getNumBytes(dstTy) > 16);
    Value val = genLoad(srcAddr, srcDataLoc, loc);

    // Apply alignment shifts when crossing the storage boundary:
    //   bytesN: left-aligned in memory/calldata, right-aligned in storage.
    //   ExtFuncRef: same left/right convention.
    unsigned alignShift = 0;
    if (sol::isBytesLikeType(dstTy))
      alignShift = (32 - sol::getNumBytes(dstTy)) * 8;
    else if (isa<sol::ExtFuncRefType>(dstTy))
      alignShift = 64;

    if (alignShift > 0) {
      if (srcDataLoc == sol::DataLocation::CallData)
        val = genCleanup(dstTy, val, loc, /*shouldRevert=*/true);

      if (!srcIsStorage && dstIsStorage)
        val =
            b.create<yul::ArithShrOp>(loc, val, bExt.genI256Const(alignShift));
      else if (srcIsStorage && !dstIsStorage)
        val =
            b.create<yul::ArithShlOp>(loc, val, bExt.genI256Const(alignShift));
    } else {
      if (dstIsStorage) {
        if (srcDataLoc == sol::DataLocation::CallData)
          val = genCleanup(dstTy, val, loc, /*shouldRevert=*/true);
        // Storage write-side: AND-mask to zero upper bits, matching old codegen
        // LValue.cpp (chopSignBits=true). genCleanup for signed ints
        // returns signextend, which would set upper bits to 1 for negative
        // values and corrupt the storage slot.
        unsigned storageBits = sol::getNumBytes(dstTy) * 8;
        APInt fieldMask = APInt::getLowBitsSet(256, storageBits);
        val = b.create<yul::AndOp>(loc, val, bExt.genI256Const(fieldMask));
      } else {
        val = genCleanup(dstTy, val, loc, srcDataLoc);
      }
    }
    genStore(val, dstAddr, dstDataLoc, loc);
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
  } else if (auto dstStructTy = dyn_cast<sol::StructType>(dstTy)) {
    // Struct copies to memory are handled by DataLocCastOpLowering.
    // genCopy only sees struct copies whose destination is storage.
    if (dstDataLoc != sol::DataLocation::Storage)
      llvm_unreachable(
          "struct -> memory copy must be lowered by DataLocCastOpLowering");
    auto srcStructTy = cast<sol::StructType>(srcTy);
    ArrayRef<Type> memberTypes = dstStructTy.getMemberTypes();
    ArrayRef<Type> srcMemberTypes = srcStructTy.getMemberTypes();
    assert(memberTypes.size() == srcMemberTypes.size());

    std::unique_ptr<StructEncodeMemberReader> srcReader =
        makeStructMemberReader(srcStructTy, b, loc, srcAddr);
    std::unique_ptr<StructMemberWriter> dstWriter =
        makeStructMemberWriter(dstStructTy, b, loc, dstAddr);

    bool srcIsCallData =
        srcStructTy.getDataLocation() == sol::DataLocation::CallData;

    for (uint64_t i = 0, e = memberTypes.size(); i < e; ++i) {
      Type dstMemberTy = memberTypes[i];
      bool packedDstMember = dstIsStorage && sol::canBePacked(dstMemberTy);
      // For storage→storage packed members, skip canonical-form conversion in
      // the reader and apply AND(fieldMask) here. This avoids the
      // signextend+AND and shl+shr round-trips that
      // genCleanupPackedStorageValue introduces for signed integers, bytesN,
      // and ExtFuncRef types.
      Value val =
          srcReader->read(i, /*skipCleanup=*/packedDstMember && srcIsStorage);
      if (packedDstMember) {
        unsigned storageBits = sol::getNumBytes(dstMemberTy) * 8;
        APInt fieldMask = APInt::getLowBitsSet(256, storageBits);
        if (srcIsStorage) {
          // Raw right-shifted field bits from the storage reader AND to field
          // width without sign-extending.
          val = b.create<yul::AndOp>(loc, val, bExt.genI256Const(fieldMask));
        } else {
          // Memory/calldata source: bytesN and ExtFuncRef are left-aligned.
          unsigned alignShift = 0;
          if (sol::isBytesLikeType(dstMemberTy))
            alignShift = (32 - sol::getNumBytes(dstMemberTy)) * 8;
          else if (isa<sol::ExtFuncRefType>(dstMemberTy))
            alignShift = 64;

          if (alignShift > 0) {
            if (srcIsCallData)
              val = genCleanup(dstMemberTy, val, loc, /*shouldRevert=*/true);
            val = b.create<yul::ArithShrOp>(loc, val,
                                            bExt.genI256Const(alignShift));
          } else if (srcIsCallData) {
            val = genCleanup(dstMemberTy, val, loc, /*shouldRevert=*/true);
            // genCleanup for signed integers returns signextend(), which leaves
            // sign bits set in bits [255..storageBits] for negative values.
            // Mask to field width so the writer receives field-masked bits.
            if (auto intTy = dyn_cast<IntegerType>(dstMemberTy);
                intTy && intTy.isSigned() && intTy.getWidth() < 256)
              val =
                  b.create<yul::AndOp>(loc, val, bExt.genI256Const(fieldMask));
          } else {
            // Memory source: AND to field width without sign-extending.
            val = b.create<yul::AndOp>(loc, val, bExt.genI256Const(fieldMask));
          }
        }
      }
      dstWriter->write(i, val, srcMemberTypes[i]);
      if (i + 1 < e) {
        srcReader->advance(i);
        dstWriter->advance(i);
      }
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

  auto shortTupleCond = bExt.genCmp(yul::CmpPredicate::slt, tupleSize,
                                    bExt.genI256Const(totCallDataHeadSz));
  genDebugRevertWithMsg(shortTupleCond, "ABI decoding: tuple data too short",
                        loc);
}

Value evm::Builder::genABIEncodingImpl(
    Type ty, Value src, Value dstAddr, ABIEncodingOptions opts,
    bool dstAddrInTail, Value tupleStart, Value tailAddr, Location loc,
    std::optional<sol::DataLocation> srcDataLoc) {
  mlir::solgen::BuilderExt bExt(b, loc);

  // Scalar
  if (sol::isScalar(ty)) {
    unsigned numBytes = sol::getNumBytes(ty);
    Value cleaned = genCleanup(ty, src, loc, srcDataLoc);
    // Right-aligned types need a left-shift in packed mode so their bytes
    // land at the low memory address after mstore.
    if (!opts.padded && !sol::isLeftAligned(ty) && numBytes < 32)
      cleaned = b.create<yul::ArithShlOp>(
          loc, cleaned, bExt.genI256Const(256 - numBytes * 8));
    b.create<yul::MStoreOp>(loc, dstAddr, cleaned);
    if (opts.dynamicInplace)
      return b.create<yul::AddOp>(
          loc, dstAddr, bExt.genI256Const(opts.padded ? 32 : numBytes));
    return tailAddr;
  }

  // String
  if (isa<sol::StringType>(ty)) {
    Value dataAddr =
        opts.dynamicInplace
            ? dstAddr
            : b.create<yul::AddOp>(loc, tailAddr, bExt.genI256Const(32));
    Value size = genCopyStringDataToMemory(src, ty, dataAddr, loc,
                                           /*withCleanup=*/true);
    if (!opts.dynamicInplace)
      b.create<yul::MStoreOp>(loc, tailAddr, size);
    Value advanceBy = opts.padded ? bExt.genRoundUpToMultiple<32>(size) : size;
    return b.create<yul::AddOp>(loc, dataAddr, advanceBy);
  }

  // Struct
  // Two emission paths:
  //   (A) packed single-cursor   -- members written contiguously.
  //   (B) standard head/tail     -- dynamic members write a tail offset in
  //                                 their head slot; static members encode
  //                                 in place at the head.
  // Members are always 32B-padded in either mode (upstream parity).
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    // ---- Shared setup: member reader + member sub-options. -------------
    auto dataLoc = structTy.getDataLocation();
    std::unique_ptr<StructEncodeMemberReader> reader =
        makeStructMemberReader(structTy, b, loc, src);

    auto memberTypes = structTy.getMemberTypes();
    ABIEncodingOptions memSub = opts;
    memSub.padded = true;

    // ---- Path (A): packed single-cursor. -------------------------------
    if (opts.dynamicInplace) {
      Value curAddr = dstAddr;
      for (uint64_t i = 0, e = memberTypes.size(); i < e; ++i) {
        Type memTy = memberTypes[i];
        Value srcVal = reader->read(i);
        curAddr = genABIEncodingImpl(memTy, srcVal, curAddr, memSub,
                                     /*dstAddrInTail=*/true,
                                     /*tupleStart=*/curAddr,
                                     /*tailAddr=*/curAddr, loc, dataLoc);
        if (i + 1 < e)
          reader->advance(i);
      }
      return curAddr;
    }

    // ---- Path (B): standard head/tail. ---------------------------------
    if (dstAddrInTail)
      tailAddr = b.create<yul::AddOp>(
          loc, dstAddr,
          bExt.genI256Const(getStructCalldataEncodedTailSize(structTy)));

    Value structHeadAddr = dstAddr;
    for (uint64_t i = 0, e = memberTypes.size(); i < e; ++i) {
      Type memTy = memberTypes[i];
      Value srcVal = reader->read(i);

      if (sol::hasDynamicallySizedElt(memTy)) {
        b.create<yul::MStoreOp>(loc, structHeadAddr,
                                b.create<yul::SubOp>(loc, tailAddr, dstAddr));
        tailAddr = genABIEncodingImpl(memTy, srcVal, tailAddr, memSub,
                                      /*dstAddrInTail=*/true, dstAddr, tailAddr,
                                      loc, dataLoc);
      } else {
        tailAddr = genABIEncodingImpl(memTy, srcVal, structHeadAddr, memSub,
                                      /*dstAddrInTail=*/false, dstAddr,
                                      tailAddr, loc, dataLoc);
      }

      if (i + 1 < e) {
        structHeadAddr = b.create<yul::AddOp>(
            loc, structHeadAddr, bExt.genI256Const(getCallDataHeadSize(memTy)));
        reader->advance(i);
      }
    }
    return tailAddr;
  }

  // Array
  // Four emission paths, in dispatch order:
  //   (A) whole-array CallDataCopy     -- scalar-element calldata array
  //                                       whose payload is 32B-padded.
  //   (B) specialized storage loops    -- value-typed storage arrays
  //                                       (compact or linear).
  //   (C) packed single-cursor loop    -- packed mode (catch-all for
  //                                       packed encoding).
  //   (D) generic head/tail loop       -- standard mode (catch-all).
  // `loadElt` reads the source element for (C) and (D); (A) and (B) read
  // the source directly.
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    Type eltTy = arrTy.getEltType();
    auto dataLoc = arrTy.getDataLocation();

    // `srcArrAddr` is assigned by each path that uses `loadElt` before
    // invoking it; `loadElt` captures it by reference.
    Value srcArrAddr, size;

    // Reads the element at `srcAddr` and yields the value passed to the
    // recursive encoder.
    auto loadElt = [&](Location loc, Value srcAddr) -> Value {
      if (dataLoc == sol::DataLocation::Memory)
        return genLoad(srcAddr, dataLoc, loc);
      if (dataLoc == sol::DataLocation::CallData) {
        if (sol::hasDynamicallySizedElt(eltTy))
          return genCalldataAccessRef(eltTy, srcArrAddr, srcAddr,
                                      /*isNonABI=*/false, loc);
        if (sol::isNonPtrRefType(eltTy))
          return srcAddr;
        return genLoad(srcAddr, dataLoc, loc);
      }
      assert(dataLoc == sol::DataLocation::Storage &&
             sol::isNonPtrRefType(eltTy));
      return srcAddr;
    };

    // ---- Shared setup: size, srcArrAddr, dstArrAddr, dstStride, tailAddr.
    // Path C doesn't use dstStride: its single-cursor loop advances by the
    // recursive call's return (variable per element), not by a fixed stride.
    if (arrTy.isDynSized()) {
      size = genDynSize(src, arrTy, loc);
      srcArrAddr = genDataAddrPtr(src, arrTy, loc);
    } else {
      size = bExt.genI256Const(arrTy.getSize());
      srcArrAddr = src;
    }

    Value dstArrAddr;
    // Length prefix: only standard-mode dynamic arrays carry one.
    if (arrTy.isDynSized() && !opts.dynamicInplace) {
      assert(dstAddr == tailAddr);
      b.create<yul::MStoreOp>(loc, dstAddr, size);
      dstArrAddr = b.create<yul::AddOp>(loc, dstAddr, bExt.genI256Const(32));
    } else {
      dstArrAddr = dstAddr;
    }

    Value dstStride = bExt.genI256Const(getCallDataHeadSize(eltTy));

    if (arrTy.isDynSized() || dstAddrInTail) {
      Value sizeInBytes = b.create<yul::MulOp>(loc, size, dstStride);
      tailAddr = b.create<yul::AddOp>(loc, dstArrAddr, sizeInBytes);
    }

    // ---- Path (A): whole-array CallDataCopy. ---------------------------
    if (dataLoc == sol::DataLocation::CallData &&
        canFastCopyCalldataArray(eltTy)) {
      if (arrTy.isDynSized()) {
        unsigned stride = getCallDataHeadSize(eltTy);
        APInt maxLength = APInt::getAllOnes(256).udiv(APInt(256, stride));
        Value tooLongCond = bExt.genCmp(yul::CmpPredicate::ugt, size,
                                        bExt.genI256Const(maxLength));
        genDebugRevertWithMsg(tooLongCond, "ABI encoding: array data too long",
                              loc);
      }
      Value sizeInBytes = b.create<yul::MulOp>(loc, size, dstStride);
      b.create<yul::CallDataCopyOp>(loc, dstArrAddr, srcArrAddr, sizeInBytes);
      return tailAddr;
    }

    // ---- Path (B): specialized storage loops. --------------------------
    if (dataLoc == sol::DataLocation::Storage && sol::isScalar(eltTy)) {
      ABIEncodingOptions sub = opts;
      sub.padded = true;
      if (sol::getNumBytes(eltTy) <= 16)
        emitCompactStorageArrayReadLoop(
            b, loc, size, dstArrAddr, srcArrAddr, dstStride,
            sol::getNumBytes(eltTy), arrTy.isDynSized(),
            [&](OpBuilder &builder, Location loc, Value slotValue,
                Value shiftBits, Value iDstAddr) {
              Value shifted =
                  builder.create<yul::ArithShrOp>(loc, slotValue, shiftBits);
              Value srcVal = genCleanupPackedStorageValue(eltTy, shifted, loc);
              genABIEncodingImpl(eltTy, srcVal, iDstAddr, sub, dstAddrInTail,
                                 tupleStart, tailAddr, loc, dataLoc);
            });
      else
        emitLinearArrayLoop(
            b, loc, size, dstArrAddr, srcArrAddr, dstStride,
            bExt.genI256Const(sol::getStorageSlotCount(eltTy)),
            [&](OpBuilder &, Location loc, Value iSrcAddr, Value iDstAddr) {
              Value srcVal = genLoad(iSrcAddr, dataLoc, loc);
              genABIEncodingImpl(eltTy, srcVal, iDstAddr, sub, dstAddrInTail,
                                 tupleStart, tailAddr, loc, dataLoc);
            });
      return tailAddr;
    }

    Value srcStride = bExt.genI256Const(getArrayEltStride(arrTy));

    // ---- Path (C): packed single-cursor loop. --------------------------
    if (opts.dynamicInplace) {
      ABIEncodingOptions sub = opts;
      sub.padded = true;
      auto forOp = bExt.createCountedLoop(
          bExt.genI256Const(0), size, bExt.genI256Const(1),
          ValueRange{dstArrAddr, srcArrAddr},
          [&](OpBuilder &nb, Location nloc, Value, ValueRange initArgs) {
            Value iDstAddr = initArgs[0];
            Value iSrcAddr = initArgs[1];
            Value srcVal = loadElt(nloc, iSrcAddr);
            // Single-cursor recursion: head/tail bookkeeping is unused, so
            // pass geometrically consistent values pinned at iDstAddr.
            Value nextDst = genABIEncodingImpl(
                eltTy, srcVal, iDstAddr, sub, /*dstAddrInTail=*/true,
                /*tupleStart=*/iDstAddr, /*tailAddr=*/iDstAddr, nloc, dataLoc);
            Value nextSrc = nb.create<yul::AddOp>(nloc, iSrcAddr, srcStride);
            return SmallVector<Value>{nextDst, nextSrc};
          });
      return forOp.getResult(1);
    }

    // ---- Path (D): generic head/tail per-element loop. -----------------
    auto forOp = bExt.createCountedLoop(
        bExt.genI256Const(0), size, bExt.genI256Const(1),
        ValueRange{dstArrAddr, srcArrAddr, tailAddr},
        [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
          Value iDstAddr = initArgs[0];
          Value iSrcAddr = initArgs[1];
          Value iTailAddr = initArgs[2];

          Value srcVal = loadElt(loc, iSrcAddr);
          Value nextTailAddr;
          ABIEncodingOptions sub = opts;
          sub.padded = true;
          if (sol::hasDynamicallySizedElt(eltTy)) {
            b.create<yul::MStoreOp>(
                loc, iDstAddr,
                b.create<yul::SubOp>(loc, iTailAddr, dstArrAddr));
            assert(dstAddrInTail);
            nextTailAddr =
                genABIEncodingImpl(eltTy, srcVal, iTailAddr, sub, dstAddrInTail,
                                   tupleStart, iTailAddr, loc, dataLoc);
          } else {
            nextTailAddr =
                genABIEncodingImpl(eltTy, srcVal, iDstAddr, sub, dstAddrInTail,
                                   tupleStart, iTailAddr, loc, dataLoc);
          }

          return SmallVector<Value>{
              b.create<yul::AddOp>(loc, iDstAddr, dstStride),
              b.create<yul::AddOp>(loc, iSrcAddr, srcStride), nextTailAddr};
        });
    return forOp.getResult(3);
  }

  llvm_unreachable("NYI");
}

Value evm::Builder::genABIEncoding(TypeRange tys, ValueRange vals,
                                   Value startAddr,
                                   std::optional<mlir::Location> locArg) {
  ABIEncodingOptions opts;
  opts.padded = true;
  opts.dynamicInplace = false;
  return genABIEncoding(tys, vals, startAddr, opts, locArg);
}

Value evm::Builder::genABIEncodingPacked(TypeRange tys, ValueRange vals,
                                         Value startAddr,
                                         std::optional<mlir::Location> locArg) {
  ABIEncodingOptions opts;
  opts.padded = false;
  opts.dynamicInplace = true;
  return genABIEncoding(tys, vals, startAddr, opts, locArg);
}

Value evm::Builder::genABIEncoding(TypeRange tys, ValueRange vals,
                                   Value startAddr, ABIEncodingOptions opts,
                                   std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  // Packed (single linear cursor): walk values sequentially.
  if (opts.dynamicInplace) {
    Value curAddr = startAddr;
    for (auto [ty, val] : llvm::zip(tys, vals))
      curAddr = genABIEncodingImpl(ty, val, curAddr, opts,
                                   /*dstAddrInTail=*/true,
                                   /*tupleStart=*/curAddr,
                                   /*tailAddr=*/curAddr, loc,
                                   /*srcDataLoc=*/std::nullopt);
    return curAddr;
  }

  // Standard mode (head/tail layout): compute head end and walk with
  // head/tail bookkeeping.
  unsigned totCallDataHeadSz = 0;
  for (Type ty : tys)
    totCallDataHeadSz += getCallDataHeadSize(ty);

  Value tupleStart = startAddr;
  Value headAddr = tupleStart;
  Value tailAddr = b.create<yul::AddOp>(loc, tupleStart,
                                        bExt.genI256Const(totCallDataHeadSz));
  for (auto it : llvm::zip(tys, vals)) {
    Type ty = std::get<0>(it);
    Value val = std::get<1>(it);
    if (sol::hasDynamicallySizedElt(ty)) {
      b.create<yul::MStoreOp>(loc, headAddr,
                              b.create<yul::SubOp>(loc, tailAddr, tupleStart));
      tailAddr = genABIEncodingImpl(ty, val, tailAddr, opts,
                                    /*dstAddrInTail=*/true, tupleStart,
                                    tailAddr, loc, /*srcDataLoc=*/std::nullopt);
    } else {
      tailAddr = genABIEncodingImpl(ty, val, headAddr, opts,
                                    /*dstAddrInTail=*/false, tupleStart,
                                    tailAddr, loc, /*srcDataLoc=*/std::nullopt);
    }
    headAddr = b.create<yul::AddOp>(loc, headAddr,
                                    bExt.genI256Const(getCallDataHeadSize(ty)));
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
  auto tailAddr = b.create<yul::AddOp>(loc, headStart, thirtyTwo);
  genStringStore(str, tailAddr, loc);
  Value stringSize = bExt.genI256Const(
      32 + mlir::solgen::getRoundUpToMultiple<32>(str.length()));

  return b.create<yul::AddOp>(loc, tailAddr, stringSize);
}

Value evm::Builder::genABITupleDecoding(Type ty, Value addr, bool fromMem,
                                        Value tupleStart, Value tupleEnd,
                                        std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  ABIDecodeGuards guards(*this, b, loc, tupleEnd);

  // TODO: Generate assertions for checking if addresses of reference types is
  // within the calldata.

  auto genLoad = [&](Value addr) -> Value {
    if (fromMem)
      return b.create<yul::MLoadOp>(loc, addr);
    return b.create<yul::CallDataLoadOp>(loc, addr);
  };

  if (isa<IntegerType>(ty) || isa<sol::EnumType>(ty) ||
      sol::isAddressLikeType(ty) || isa<sol::FixedBytesType>(ty) ||
      isa<sol::ExtFuncRefType>(ty)) {
    Value arg = genLoad(addr);
    return genCleanup(ty, arg, loc, /*shouldRevert=*/true);
  }

  // Array type
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    Value thirtyTwo = bExt.genI256Const(32);
    Type eltTy = arrTy.getEltType();

    if (arrTy.getDataLocation() == sol::DataLocation::CallData) {
      if (!arrTy.isDynSized()) {
        guards.requireFixedArraySpan(addr, arrTy);
        return addr;
      }

      guards.requireArrayOffsetInBounds(addr);

      Value i256Size = genLoad(addr);
      Value srcAddr = b.create<yul::AddOp>(loc, addr, thirtyTwo);
      guards.requireUint64(i256Size, ABIDecodeGuards::kInvalidArrayLength);
      guards.requireDynamicArraySpan(srcAddr, i256Size, eltTy);
      return bExt.genLLVMStruct({srcAddr, i256Size});
    }

    guards.requireArrayOffsetInBounds(addr);

    Value dstAddr, srcAddr, size, ret;
    if (arrTy.isDynSized()) {
      Value i256Size = genLoad(addr);
      srcAddr = b.create<yul::AddOp>(loc, addr, thirtyTwo);
      guards.requireDynamicArraySpan(srcAddr, i256Size, eltTy);

      dstAddr = genMemAllocForDynArray(
          i256Size, b.create<yul::MulOp>(loc, i256Size, thirtyTwo), loc, true);
      ret = dstAddr;
      // Skip the size fields in both the addresses.
      dstAddr = b.create<yul::AddOp>(loc, dstAddr, thirtyTwo);
      size = i256Size;
    } else {
      dstAddr = genMemAlloc(bExt.genI256Const(arrTy.getSize() * 32), loc);
      ret = dstAddr;
      srcAddr = addr;
      size = bExt.genI256Const(arrTy.getSize());
      guards.requireFixedArraySpan(srcAddr, arrTy);
    }

    bExt.createCountedLoop(
        bExt.genI256Const(0), size, bExt.genI256Const(1),
        ValueRange{dstAddr, srcAddr},
        [&](OpBuilder &b, Location loc, Value, ValueRange initArgs) {
          Value iDstAddr = initArgs[0];
          Value iSrcAddr = initArgs[1];

          if (sol::hasDynamicallySizedElt(eltTy)) {
            Value innerOffset = genLoad(iSrcAddr);
            guards.requireUint64(innerOffset,
                                 ABIDecodeGuards::kInvalidArrayOffset);

            // The elements are offset wrt to the start of this array (after the
            // size field if dynamic) that contain the inner element.
            Value offsetFromSrcArr =
                b.create<yul::AddOp>(loc, srcAddr, innerOffset);
            b.create<yul::MStoreOp>(loc, iDstAddr,
                                    genABITupleDecoding(eltTy, offsetFromSrcArr,
                                                        fromMem, tupleStart,
                                                        tupleEnd, loc));
          } else {
            b.create<yul::MStoreOp>(loc, iDstAddr,
                                    genABITupleDecoding(eltTy, iSrcAddr,
                                                        fromMem, tupleStart,
                                                        tupleEnd, loc));
          }

          Value srcStride = bExt.genI256Const(getCallDataHeadSize(eltTy));
          return SmallVector<Value>{
              b.create<yul::AddOp>(loc, iDstAddr, bExt.genI256Const(32)),
              b.create<yul::AddOp>(loc, iSrcAddr, srcStride)};
        });
    return ret;
  }

  // String type
  if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
    guards.requireArrayOffsetInBounds(addr);

    Value tailAddr = addr;
    Value sizeInBytes = genLoad(tailAddr);
    Value thirtyTwo = bExt.genI256Const(32);
    Value srcDataAddr = b.create<yul::AddOp>(loc, tailAddr, thirtyTwo);
    Value endAddr = b.create<yul::AddOp>(loc, srcDataAddr, sizeInBytes);
    guards.requireEndInBounds(endAddr,
                              ABIDecodeGuards::kInvalidByteArrayLength);

    if (stringTy.getDataLocation() == sol::DataLocation::CallData)
      return bExt.genLLVMStruct({srcDataAddr, sizeInBytes});

    // Copy the decoded string to a new memory allocation.
    Value dstAddr = genMemAllocForDynArray(
        sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes), loc, true);
    Value dstDataAddr = b.create<yul::AddOp>(loc, dstAddr, thirtyTwo);

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
    Value cleanupAddr = b.create<yul::AddOp>(loc, dstDataAddr, sizeInBytes);
    b.create<yul::MStoreOp>(loc, cleanupAddr, bExt.genI256Const(0));

    return dstAddr;
  }

  // Struct type
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    auto dataLoc = structTy.getDataLocation();
    assert((dataLoc == sol::DataLocation::CallData ||
            dataLoc == sol::DataLocation::Memory) &&
           "Unexpected struct data location");

    Value srcEnd = b.create<yul::AddOp>(
        loc, addr,
        bExt.genI256Const(getStructCalldataEncodedTailSize(structTy)));
    guards.requireEndInBounds(srcEnd, guards.getStructHeadTooShortMsg(dataLoc));

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
        guards.requireUint64(tailOffset, ABIDecodeGuards::kInvalidStructOffset);

        Value tailAddr = b.create<yul::AddOp>(loc, addr, tailOffset);
        memberVal =
            genABITupleDecoding(memTy, tailAddr, fromMem, addr, tupleEnd, loc);
      } else {
        memberVal = genABITupleDecoding(memTy, srcHeadAddr, fromMem, addr,
                                        tupleEnd, loc);
      }

      b.create<yul::MStoreOp>(loc, dstHeadAddr, memberVal);
      srcHeadAddr = b.create<yul::AddOp>(
          loc, srcHeadAddr, bExt.genI256Const(getCallDataHeadSize(memTy)));
      dstHeadAddr =
          b.create<yul::AddOp>(loc, dstHeadAddr, bExt.genI256Const(32));
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
  ABIDecodeGuards guards(*this, b, loc, tupleEnd);

  // TODO? {en|de}codingType() for sol dialect types.

  genABITupleSizeAssert(tys, b.create<yul::SubOp>(loc, tupleEnd, tupleStart),
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
      guards.requireUint64(tailOffset, ABIDecodeGuards::kInvalidTupleOffset);

      Value tailAddr = b.create<yul::AddOp>(loc, tupleStart, tailOffset);
      results.push_back(genABITupleDecoding(ty, tailAddr, fromMem, tupleStart,
                                            tupleEnd, loc));
    } else {
      results.push_back(genABITupleDecoding(ty, headAddr, fromMem, tupleStart,
                                            tupleEnd, loc));
    }
    headAddr = b.create<yul::AddOp>(loc, headAddr,
                                    bExt.genI256Const(getCallDataHeadSize(ty)));
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

  auto ifOp = bExt.createIf(cond);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
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
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = bExt.createIf(cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genForwardingRevert(loc);
}

void evm::Builder::genRevert(std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);
  mlir::Value zero = bExt.genI256Const(0);
  b.create<yul::RevertOp>(loc, zero, zero);
}

void evm::Builder::genRevert(Value cond, std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = bExt.createIf(cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevert(loc);
}

void evm::Builder::genRevert(TypeRange tys, ValueRange vals,
                             StringRef signature,
                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value selectorAddr = genFreePtr(loc);
  b.create<yul::MStoreOp>(loc, selectorAddr, bExt.genI256Selector(signature));
  Value tupleStart =
      b.create<yul::AddOp>(loc, selectorAddr, bExt.genI256Const(4));
  Value tupleEnd = genABIEncoding(tys, vals, tupleStart, loc);
  Value size = b.create<yul::SubOp>(loc, tupleEnd, selectorAddr);
  b.create<yul::RevertOp>(loc, selectorAddr, size);
}

void evm::Builder::genRevert(Value cond, TypeRange tys, ValueRange vals,
                             StringRef signature,
                             std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = bExt.createIf(cond);
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
      b.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(4));
  Value tailAddr =
      genABITupleEncoding(msg, /*headStart=*/freePtrPostSelector, loc);

  // Generate the revert.
  auto retDataSize = b.create<yul::SubOp>(loc, tailAddr, freePtr);
  b.create<yul::RevertOp>(loc, freePtr, retDataSize);
}

void evm::Builder::genRevertWithMsg(Value cond, std::string const &msg,
                                    std::optional<mlir::Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = bExt.createIf(cond);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genRevertWithMsg(msg, loc);
}

void evm::Builder::genDebugRevertWithMsg(Value cond, std::string const &msg,
                                         std::optional<mlir::Location> locArg) {
  if (sol::shouldEmitDebugRevertStrings(mod) && !msg.empty())
    genRevertWithMsg(cond, msg, locArg);
  else
    genRevert(cond, locArg);
}

void evm::Builder::genUserRevertWithMsg(std::string const &msg,
                                        std::optional<mlir::Location> locArg) {
  if (sol::shouldKeepUserRevertStrings(mod))
    genRevertWithMsg(msg, locArg);
  else
    genRevert(locArg);
}

void evm::Builder::genUserRevertWithMsg(Value cond, std::string const &msg,
                                        std::optional<mlir::Location> locArg) {
  if (sol::shouldKeepUserRevertStrings(mod))
    genRevertWithMsg(cond, msg, locArg);
  else
    genRevert(cond, locArg);
}

void evm::Builder::genDbgRevert(ValueRange vals,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  Value freePtr = genFreePtr(loc);
  unsigned retDataSize = 0;
  for (Value val : vals) {
    auto offset =
        b.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(retDataSize));
    b.create<yul::MStoreOp>(loc, offset, val);
    retDataSize += 32;
  }
  b.create<yul::RevertOp>(loc, freePtr, bExt.genI256Const(retDataSize));
}

void evm::Builder::genCondDbgRevert(Value cond, ValueRange vals,
                                    std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;
  mlir::solgen::BuilderExt bExt(b, loc);

  auto ifOp = bExt.createIf(cond);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(&ifOp.getThenRegion().front());
  genDbgRevert(vals, loc);
}

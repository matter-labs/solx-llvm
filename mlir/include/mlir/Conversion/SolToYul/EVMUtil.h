//===- EVMUtil.h - EVM specific MLIR utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EVM-specific MLIR utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOYUL_EVMUTIL_H
#define MLIR_CONVERSION_SOLTOYUL_EVMUTIL_H

#include "mlir/Conversion/SolToYul/EVMConstants.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace mlir {
namespace evm {

enum AddrSpace : unsigned {
  AddrSpace_Stack = 0,
  AddrSpace_Heap = 1,
  AddrSpace_CallData = 2,
  AddrSpace_ReturnData = 3,
  AddrSpace_Code = 4,
  AddrSpace_Storage = 5,
  AddrSpace_TransientStorage = 6,
};

enum ByteLen {
  ByteLen_Byte = 1,
  ByteLen_X32 = 4,
  ByteLen_X64 = 8,
  ByteLen_EthAddr = 20,
  ByteLen_Field = 32
};

using AllocSize = int64_t;

/// Returns the alignment of `addrSpace`
unsigned getAlignment(AddrSpace addrSpace);
/// Returns the alignment of the LLVMPointerType value `ptr`
unsigned getAlignment(Value ptr);

// FIXME: Remove this! The lowering should not expand the ABI encoding/decoding
// code (where this is used) involved in sol.contract, sol.emit etc. lowering.
// Instead it should generate a high level sol op that can have a custom
// lowering for each target.
/// MLIR version of solidity ast's Type::calldataHeadSize.
unsigned getCallDataHeadSize(Type ty);

/// Per-element iteration step. Unit depends on data location: memory-word
/// bytes for Memory/CallData, storage slots for Storage.
unsigned getArrayEltStride(sol::ArrayType arrTy);

/// Returns the size (in bytes) of static type without recursively calculating
/// the element type size.
int64_t getMallocSize(Type ty);

/// ABI encoding knobs. Trimmed subset of upstream solc's `EncodingOptions` in
/// libsolidity/codegen/ABIFunctions.h:129-146 - only the two fields any type
/// branch currently consults are kept. The two common shapes
/// (`{padded=true, dynamicInplace=false}` for standard, `{padded=false,
/// dynamicInplace=true}` for packed) have wrappers (`genABIEncoding` /
/// `genABIEncodingPacked`); only uncommon callers spell out the struct.
///
/// Naming convention used in the implementation comments:
///  - "standard mode" / "standard encoding" => `dynamicInplace=false`. What
///    `abi.encode` produces: head + tail with offsets, length prefixes on
///    dynamic types.
///  - "packed mode" / "packed encoding"     => `dynamicInplace=true`. What
///    `abi.encodePacked` produces: single contiguous cursor, no offsets, no
///    length prefixes. `padded` varies within packed mode - false at the top
///    level, true inside containers after the recursive call site resets it
///    (upstream parity, see note below).
///
/// Container-element / struct-member recursion always forces `padded=true`,
/// mirroring upstream solc's `subOptions` reset in
/// `abiEncodingFunctionSimpleArray`, `abiEncodingFunctionCompactStorageArray`,
/// and `abiEncodingFunctionStruct`. This is spelled out at each recursive
/// call site rather than wrapped in a helper, since some sites additionally
/// force `dynamicInplace=false` (head/tail element calls) while others
/// preserve `dynamicInplace=true` (packed struct/array recursion).
struct ABIEncodingOptions {
  /// Pad value types to 32B (no MSB shift); zero-pad `bytes`/`string` payload
  /// to a 32B multiple; advance head by padded `calldataEncodedSize`.
  bool padded = true;
  /// Drop length prefix for dynamic arrays/`bytes`/`string`; encode
  /// struct/array body contiguously (no head + tail with offset words).
  bool dynamicInplace = false;
};

/// IR Builder for EVM specific lowering.
class Builder {
  // It's possible to provide a mlirgen::BuilderHelper member with same default
  // location, but then it becomes tricky to keep the default location behaviour
  // consistent.

  ModuleOp mod;
  OpBuilder &b;
  Location defLoc;

public:
  explicit Builder(ModuleOp mod, OpBuilder &b, Location loc)
      : mod(mod), b(b), defLoc(loc) {
    assert(mod && "expected attached module for EVM builder");
  }

  /// Generates a pointer to the address in the heap.
  Value genHeapPtr(Value addr, std::optional<Location> locArg = std::nullopt);

  /// Generates a pointer to the address in the calldata.
  Value genCallDataPtr(Value addr,
                       std::optional<Location> locArg = std::nullopt);

  /// Generates a pointer to the address in the returndata.
  Value genReturnDataPtr(Value addr,
                         std::optional<Location> locArg = std::nullopt);

  /// Generates a pointer to the address in the storage.
  Value genStoragePtr(Value addr,
                      std::optional<Location> locArg = std::nullopt);

  /// Generates a pointer to the address in the transient storage.
  Value genTStoragePtr(Value addr,
                       std::optional<Location> locArg = std::nullopt);

  /// Generates a pointer to the address in the code.
  Value genCodePtr(Value addr, std::optional<Location> locArg = std::nullopt);

  //
  // The following APIs are used in the stage1 lowering.
  //
  // TODO? Some of these could be transformed to mlir ops instead. This would
  // allow targets to define custom lowering. For instance, some targets might
  // not need the `type(uint64).max` check in the memory allocation code
  // generated by the genMemAlloc APIs.
  //

  /// Generates the free pointer.
  ///
  /// NOTE! Watch out for free-ptr loads between this and the update!
  Value genFreePtr(std::optional<Location> locArg = std::nullopt);

  /// Generates the free pointer update code.
  void genFreePtrUpd(Value freePtr, Value size,
                     std::optional<Location> locArg = std::nullopt);

  /// Generates the memory allocation code.
  Value genMemAlloc(Value size, std::optional<Location> locArg = std::nullopt);
  Value genMemAlloc(AllocSize size,
                    std::optional<Location> locArg = std::nullopt);

  /// Generates the memory allocation code for dynamic array.
  Value genMemAllocForDynArray(Value sizeVar, Value sizeInBytes,
                               std::optional<Location> locArg = std::nullopt,
                               bool genLengthPanicGuard = false);

  /// Generates the memory allocation code.
  Value genMemAlloc(Type ty, bool zeroInit, ValueRange initVals, Value sizeVar,
                    Type sizeVarTy,
                    std::optional<Location> locArg = std::nullopt);

  /// Cleans up a promoted Solidity integer value according to width/sign.
  Value genIntCleanup(unsigned width, bool isSigned, Value val,
                      std::optional<Location> locArg = std::nullopt);

  /// Cleans up a Solidity scalar value. Scalar validation failures use
  /// ABI-style revert when \p shouldRevert is set; otherwise cleanup-time enum
  /// validation uses panic.
  Value genCleanup(Type ty, Value val,
                   std::optional<Location> locArg = std::nullopt,
                   bool shouldRevert = false);
  Value genCleanup(Type ty, Value val, std::optional<Location> locArg,
                   std::optional<sol::DataLocation> srcDataLoc);

private:
  Value genMemAlloc(Type ty, bool zeroInit, ValueRange initVals, Value sizeVar,
                    int64_t recDepth,
                    std::optional<Location> locArg = std::nullopt);

  /// Recursive primitive for `genABIEncoding` covering the full type taxonomy
  /// parameterised by `opts`. `dstAddrInTail`, `tupleStart`, `tailAddr` are
  /// meaningful only when `opts.dynamicInplace=false`.
  Value genABIEncodingImpl(Type ty, Value src, Value dstAddr,
                           ABIEncodingOptions opts, bool dstAddrInTail,
                           Value tupleStart, Value tailAddr, Location loc,
                           std::optional<sol::DataLocation> srcDataLoc);

  /// Zeroes storage slots in the out-of-place data area of a storage
  /// string/bytes at \p dstAddr that are no longer needed when the content
  /// changes from \p oldLength to \p newLength bytes.  Called only when
  /// \p oldLength > 31 (i.e. the old value used out-of-place storage).
  void genClearStringStorageTail(
      mlir::Value dstAddr, mlir::Value oldLength, mlir::Value newLength,
      std::optional<mlir::Location> locArg = std::nullopt);

  /// Returns the number of storage slots occupied by \p len elements of type
  /// \p eltTy: ceil(len / elemsPerSlot) for packable types, len * slotsPerElt
  /// for non-packable types.
  mlir::Value
  genStorageArraySlotCount(mlir::Value len, mlir::Type eltTy,
                           std::optional<mlir::Location> locArg = std::nullopt);

  /// Resizes a dynamic storage array: writes \p newLen to \p arraySlot,
  /// then zeroes any storage slots that fall outside the new range if the
  /// array shrank. Panics (ResourceError) if \p newLen exceeds
  /// type(uint64).max.
  void
  genResizeDynStorageArray(mlir::Value arraySlot, mlir::Value newLen,
                           mlir::Type eltTy,
                           std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates a revert with message.
  void genRevertWithMsg(std::string const &msg,
                        std::optional<Location> locArg = std::nullopt);
  void genRevertWithMsg(Value cond, std::string const &msg,
                        std::optional<Location> locArg = std::nullopt);

public:
  //
  // TODO? Should we work with the high level types + OpAdaptor for the APIs
  // that work with low level integral type pointers?
  //

  /// Generates a low level integral type pointer to the address holding the
  /// data of a dynamic allocation.
  Value genDataAddrPtr(Value addr, Type ty,
                       std::optional<Location> locArg = std::nullopt);
  Value genDataAddrPtr(Value addr, sol::DataLocation dataLoc,
                       std::optional<Location> locArg = std::nullopt);

  /// Extracts the leading fixed-bytes value from a dynamic bytes type.
  Value genDynBytesToFixedBytes(Value src, Type srcTy,
                                sol::FixedBytesType dstTy,
                                std::optional<Location> locArg = std::nullopt);

  /// Generates the address computation of the array or string at index.
  // TODO: Data-location should be fetched from the type! Implement APIs to
  // "clone" reference types with different data-locations.
  Value genAddrAtIdx(Value baseAddr, Value idx, Type ty,
                     sol::DataLocation dataLoc,
                     std::optional<Location> locArg = std::nullopt);

  /// Resolves a calldata relative-offset pointer (\p ptr) to the effective
  /// address of a dynamically-encoded ABI element, with bounds validation.
  /// \p baseAddr is the absolute start of the enclosing calldata head section.
  ///
  /// - Dynamic arrays / strings / bytes: returns {dataPtr, length} fat pointer.
  /// - Fixed-size arrays with dynamic children: returns the resolved absolute
  ///   address (baseAddr + relOffset).
  /// - Nested calldata structs: forwarded by their head address.
  ///
  /// When \p isNonABI is false (default, ABI-function context) the revert
  /// messages read "Invalid calldata access {offset,length,stride}".
  /// When \p isNonABI is true (non-ABI / internal calldata parameter) they
  /// read "Invalid calldata tail {offset,length}" and "Calldata tail too
  /// short".
  Value genCalldataAccessRef(mlir::Type ty, Value baseAddr, Value ptr,
                             bool isNonABI = false,
                             std::optional<Location> locArg = std::nullopt);

  /// Generates {slot, offset} for packed storage array indexing.
  Value genPackedStorageAddr(Value baseSlot, Value idx, Type eltTy,
                             bool isDataLeftAligned = false,
                             std::optional<Location> locArg = std::nullopt);

  /// Returns {slot, byteOffset} for an element at \p idx in a packed storage
  /// array rooted at \p baseSlot, where elements are \p eltByteSize bytes each.
  std::pair<Value, Value>
  genPackedStorageAddrPair(Value baseSlot, Value idx, unsigned eltByteSize,
                           bool isDataLeftAligned = false,
                           std::optional<Location> locArg = std::nullopt);

  /// Loads slot and punches hole: and(sload/tload(slot), holeMask)
  /// where holeMask = not(ones(numBits) << shiftBits)
  Value genPunchHole(Value slot, Value shiftBits, unsigned numBits,
                     sol::DataLocation dataLoc = sol::DataLocation::Storage,
                     std::optional<Location> locArg = std::nullopt);

  /// Cleans up a packed storage value to match Solidity storage-load semantics
  /// for the given element type.
  Value
  genCleanupPackedStorageValue(Type eltTy, Value value,
                               std::optional<Location> locArg = std::nullopt);

  /// Inserts integer value (<=32 bytes) to the slot value:
  /// or(and(slot, holeMask), shiftedVal), where
  /// holeMask = not(ones(numBits) << offset * 8),
  /// shiftedVal = (intVal << offset * 8)
  Value genInsertIntToSlot(Value slot, Value offset, Value intVal,
                           unsigned numBits, std::optional<Location> locArg);

  /// Generates a load from the low level integral type address.
  Value genLoad(Value addr, sol::DataLocation dataLoc,
                std::optional<Location> locArg = std::nullopt);

  /// Generates a size load from addr of dynamic type.
  Value genDynSize(Value addr, Type ty,
                   std::optional<Location> locArg = std::nullopt);

  /// Generates a store to the low level integral type address.
  void genStore(Value val, Value addr, sol::DataLocation dataLoc,
                std::optional<Location> locArg = std::nullopt);

  /// Generates the store of string at address.
  void genStringStore(std::string const &str, Value addr,
                      std::optional<Location> locArg = std::nullopt);

  /// Generates length of a string in storage.
  mlir::Value
  genStorageStringLength(mlir::Value lengthSlot,
                         std::optional<mlir::Location> locArg = std::nullopt);

  /// Low-level helper: copies string data from storage into an
  /// already-allocated memory area \p dstDataAddr, which is the destination
  /// for the string's raw bytes. For length-prefixed encodings in memory,
  /// this is typically the address immediately past the length word. Uses the
  /// pre-decoded \p lengthSlot and \p length. Does not write the length word.
  void genCopyStringDataFromStorageToMemory(
      mlir::Value src, mlir::Value lengthSlot, mlir::Value length,
      mlir::Value dstDataAddr,
      std::optional<mlir::Location> locArg = std::nullopt);

  /// Copies string data from \p src (any data location encoded in \p ty) into
  /// the already-allocated memory area \p dstDataAddr (the address past the
  /// length word). Does not write the length word. Returns the byte length.
  /// When \p withCleanup is true, emits a trailing zero-word cleanup store at
  /// dstDataAddr + length.
  mlir::Value
  genCopyStringDataToMemory(mlir::Value src, mlir::Type ty,
                            mlir::Value dstDataAddr,
                            std::optional<mlir::Location> locArg = std::nullopt,
                            bool withCleanup = false);

  /// Zeroes storage elements [\p startIdx, \p endIdx) of the storage array
  /// \p arrTy at \p arraySlot.  For a dynamic array, \p arraySlot is the
  /// length slot and the data area is keccak256(\p arraySlot).  For a static
  /// array, \p arraySlot is also the base of the data area.
  ///
  /// All Solidity storage element types are handled recursively:
  ///  - Scalars and fixed-size arrays of scalars: all occupied slots zeroed
  ///    directly.
  ///  - Packed types (elemsPerSlot >= 2, e.g. bool, uint8): partial-slot
  ///    masking is applied at the low boundary when \p startIdx does not fall
  ///    on a slot boundary.
  ///  - Strings/bytes: out-of-place data area cleared, then the length slot
  ///    zeroed.
  ///  - Dynamic sub-arrays: length slot zeroed, data area cleared recursively.
  ///  - Fixed-size arrays with dynamically-sized element types: each element
  ///    cleared recursively.
  ///  - Structs: each member cleared recursively; packed members sharing a
  ///    storage slot are deduplicated.
  ///
  /// Set \p isDecrement when the caller is popping exactly one element, i.e.
  /// \p endIdx == \p startIdx + 1.  Two optimisations are applied:
  ///  1. The range-guard branch (\p startIdx < \p endIdx) is omitted.
  ///  2. For non-packed element types the loop is unrolled into
  ///     \c getStorageSlotCount(eltTy) direct stores (a compile-time constant).
  void
  genClearStorageArrayTail(mlir::Value arraySlot, mlir::sol::ArrayType arrTy,
                           mlir::Value startIdx, mlir::Value endIdx,
                           bool isDecrement = false,
                           std::optional<mlir::Location> locArg = std::nullopt);

  /// Copies a string from \p src (any data location encoded in \p ty) to the
  /// storage slot \p dstAddr, handling in-place / out-of-place encoding and
  /// zeroing storage slots no longer needed by the new value.
  void
  genCopyStringToStorage(mlir::Value src, mlir::Type ty, mlir::Value dstAddr,
                         std::optional<mlir::Location> locArg = std::nullopt);

  /// Copies an object of type \p ty from \p srcAddr to \p dstAddr.
  ///
  /// Currently supports only arrays whose leaf element type is either
  /// \c StringType or an integer that occupies a full 32-byte storage slot.
  void genCopy(mlir::Type srcTy, mlir::Type dstTy, mlir::Value srcAddr,
               mlir::Value dstAddr, mlir::sol::DataLocation srcDataLoc,
               mlir::sol::DataLocation dstDataLoc,
               std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the 'push' of a value to string.
  void genPushToString(mlir::Value srcAddr, mlir::Value value,
                       std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the 'push' of a default value to string and
  /// returns a fat pointer to the newly added element.
  Value
  genPushVoidToString(Value srcAddr,
                      std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates the 'pop' for string.
  void genPopString(mlir::Value srcAddr, mlir::Value oldData,
                    mlir::Value length,
                    std::optional<mlir::Location> locArg = std::nullopt);

  /// Generates {slot, offset} for string storage indexing.
  Value genStringItemAddress(mlir::Value srcAddr, mlir::Value idx,
                             std::optional<Location> locArg = std::nullopt);

  /// Generates an assertion that the tuple size should be less than `size`.
  void genABITupleSizeAssert(TypeRange tys, Value size,
                             std::optional<Location> locArg = std::nullopt);

  /// Encodes `vals` of `tys` into memory at `startAddr`. Returns the address
  /// just past the last written byte. Singleton ranges are valid (e.g.
  /// indexed-event keccak input).
  Value genABIEncoding(TypeRange tys, ValueRange vals, Value startAddr,
                       ABIEncodingOptions opts,
                       std::optional<Location> locArg = std::nullopt);

  /// Standard ABI encoding (`abi.encode` shape).
  Value genABIEncoding(TypeRange tys, ValueRange vals, Value startAddr,
                       std::optional<Location> locArg = std::nullopt);

  /// Packed ABI encoding (`abi.encodePacked` / indexed-event keccak input).
  Value genABIEncodingPacked(TypeRange tys, ValueRange vals, Value startAddr,
                             std::optional<Location> locArg = std::nullopt);

  Value genABITupleEncoding(std::string const &str, Value headStart,
                            std::optional<Location> locArg = std::nullopt);

  Value genABITupleDecoding(Type ty, Value addr, bool fromMem, Value tupleStart,
                            Value tupleEnd,
                            std::optional<Location> locArg = std::nullopt);

  /// Generates the tuple decoder code as per the ABI and populates the results.
  void genABITupleDecoding(TypeRange tys, Value tupleStart, Value tupleEnd,
                           std::vector<Value> &results, bool fromMem,
                           std::optional<Location> locArg = std::nullopt);

  /// Generates the panic code.
  void genPanic(PanicCode code, std::optional<Location> locArg = std::nullopt);
  void genPanic(PanicCode code, Value cond,
                std::optional<Location> locArg = std::nullopt);

  /// Generates a revert with message if compiler-generated debug revert strings
  /// should be emitted, otherwise generates a revert without message.
  void genDebugRevertWithMsg(Value cond, std::string const &msg,
                             std::optional<Location> locArg = std::nullopt);

  /// Generates a revert with message if user-supplied revert strings should be
  /// preserved, otherwise generates a revert without message.
  void genUserRevertWithMsg(std::string const &msg,
                            std::optional<Location> locArg = std::nullopt);
  void genUserRevertWithMsg(Value cond, std::string const &msg,
                            std::optional<Location> locArg = std::nullopt);

  /// Generates a forwarding revert.
  void genForwardingRevert(std::optional<Location> locArg = std::nullopt);
  void genForwardingRevert(Value cond,
                           std::optional<Location> locArg = std::nullopt);

  /// Generates a revert(0, 0) unconditionally.
  void genRevert(std::optional<Location> locArg = std::nullopt);

  /// Generates a revert without message.
  void genRevert(Value cond, std::optional<Location> locArg = std::nullopt);

  /// Generates a revert with the abi encoded args.
  void genRevert(Value cond, TypeRange tys, ValueRange vals,
                 StringRef signature,
                 std::optional<Location> locArg = std::nullopt);
  void genRevert(TypeRange tys, ValueRange vals, StringRef signature,
                 std::optional<Location> locArg = std::nullopt);

  /// Generates a revert with values. (Useful for debugging)
  void genDbgRevert(ValueRange vals,
                    std::optional<Location> locArg = std::nullopt);

  /// Conditionally generates a revert with values. (Useful for debugging)
  void genCondDbgRevert(Value cond, ValueRange vals,
                        std::optional<Location> locArg = std::nullopt);
};

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOYUL_EVMUTIL_H

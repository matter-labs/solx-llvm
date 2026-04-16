//===- Sol.h - MLIR Solidity dialect ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Solidity dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_SOL_H_
#define MLIR_DIALECT_SOL_SOL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Sol/SolInterfaces.h.inc"
#include "mlir/Dialect/Sol/SolOpsDialect.h.inc"
#include "mlir/Dialect/Sol/SolOpsEnums.h.inc"

namespace mlir {
namespace sol {

/// Returns the data-location of type.
DataLocation getDataLocation(Type ty);
using Index = uint64_t;

/// sol dialect version of solc's Type::leftAligned().
bool isLeftAligned(Type ty);

/// Returns true if the type is dynamically sized.
bool isDynamicallySized(Type ty);

/// Returns true if the type or its element or member type (recursively) is
/// dynamically sized.
bool hasDynamicallySizedElt(Type ty);

/// Returns true if the type is a reference type (not exactly solidity's
/// reference types).
bool isRefType(Type ty);

/// Returns true if the type is a reference type but not a pointer type.
bool isNonPtrRefType(Type ty);

/// Returns the element type of a non mapping reference type.
Type getEltType(Type ty, Index structTyIdx = 0);

/// Returns true if compiler-generated debug revert strings should be emitted.
bool shouldEmitDebugRevertStrings(ModuleOp mod);

/// Returns true if user-supplied revert strings should be preserved.
bool shouldKeepUserRevertStrings(ModuleOp mod);

/// Return true if the type is address-like (i.e. address or contract type).
bool isAddressLikeType(Type ty);

/// Return true if the type is bytes-like (i.e. fixedbytes or byte).
bool isBytesLikeType(Type ty);

/// Return the byte size of a bytes-like type.
unsigned getBytesSize(Type ty);

/// MLIR version of solidity ast's Type::storageSize().
unsigned getStorageSlotCount(Type ty);

/// Returns true if the type can be packed within a storage slot.
/// Packable types (scalars) need {slot, offset} representation.
/// Non-packable types (arrays, structs, mappings) are slot-aligned and only
/// need slot.
bool canBePacked(mlir::Type ty);

/// Returns the byte size of a packable type in storage.
unsigned getStorageByteSize(mlir::Type ty);

///
/// The following functions are used to query the capabilities of the specified
/// evm in the module.
///
bool evmhasStaticCall(ModuleOp mod);
bool evmSupportsReturnData(ModuleOp mod);
bool evmCanOverchargeGasForCall(ModuleOp mod);

struct StackResource : public SideEffects::Resource::Base<StackResource> {
  StringRef getName() final { return "<Stack>"; }
};

struct CallDataResource : public SideEffects::Resource::Base<CallDataResource> {
  StringRef getName() final { return "<CallData>"; }
};

struct MemoryResource : public SideEffects::Resource::Base<MemoryResource> {
  StringRef getName() final { return "<Memory>"; }
};

struct StorageResource : public SideEffects::Resource::Base<StorageResource> {
  StringRef getName() final { return "<Storage>"; }
};

struct TransientResource
    : public SideEffects::Resource::Base<TransientResource> {
  StringRef getName() final { return "<Transient>"; }
};

struct ImmutableResource
    : public SideEffects::Resource::Base<ImmutableResource> {
  StringRef getName() final { return "<Immutable>"; }
};

struct ValidationResource
    : public SideEffects::Resource::Base<ValidationResource> {
  StringRef getName() final { return "<Validation>"; }
};

mlir::SideEffects::Resource *getResource(DataLocation dataLoc);

} // namespace sol
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Sol/SolOpsAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Sol/SolOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Sol/SolOps.h.inc"

#endif // MLIR_DIALECT_SOL_SOL_H_

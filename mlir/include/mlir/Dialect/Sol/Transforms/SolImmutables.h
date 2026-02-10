//===- SolImmutables.h - Solidity immutable lowering ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers for lowering Solidity immutables.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_TRANSFORMS_SOLIMMUTABLES_H
#define MLIR_DIALECT_SOL_TRANSFORMS_SOLIMMUTABLES_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <cstdint>

namespace mlir {
namespace evm {

/// Lowers the llvm.setimmutable ops (within llvm dialect).
void lowerSetImmutables(ModuleOp mod,
                        llvm::StringMap<SmallVector<uint64_t>> immMap);

/// Removes llvm.setimmutable ops.
void removeSetImmutables(ModuleOp mod);

} // namespace evm
} // namespace mlir

#endif // MLIR_DIALECT_SOL_TRANSFORMS_SOLIMMUTABLES_H

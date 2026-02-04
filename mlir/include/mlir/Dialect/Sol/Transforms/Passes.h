//===- Passes.h - Solidity dialect transform passes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the transformation passes for the Solidity dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_SOL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace sol {

/// Creates a pass that lowers modifier-related ops.
std::unique_ptr<Pass> createModifierOpLoweringPass();

/// Creates a pass for loop invariant code motion.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

} // namespace sol
} // namespace mlir

#endif // MLIR_DIALECT_SOL_TRANSFORMS_PASSES_H

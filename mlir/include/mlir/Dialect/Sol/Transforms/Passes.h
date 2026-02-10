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

namespace mlir {
namespace sol {

#define GEN_PASS_DECL
#include "mlir/Dialect/Sol/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Sol/Transforms/Passes.h.inc"

} // namespace sol
} // namespace mlir

#endif // MLIR_DIALECT_SOL_TRANSFORMS_PASSES_H

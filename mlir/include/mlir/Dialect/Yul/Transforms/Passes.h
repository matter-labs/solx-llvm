//===- Passes.h - Yul dialect transform passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the transformation passes for the Yul dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_YUL_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_YUL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace yul {

#define GEN_PASS_DECL
#include "mlir/Dialect/Yul/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Yul/Transforms/Passes.h.inc"

} // namespace yul
} // namespace mlir

#endif // MLIR_DIALECT_YUL_TRANSFORMS_PASSES_H

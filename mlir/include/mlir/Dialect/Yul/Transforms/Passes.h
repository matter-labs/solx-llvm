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
#include <memory>

namespace mlir {
namespace yul {

/// Creates a pass that fuses free pointer operations.
std::unique_ptr<Pass> createFuseFreePtrPass();

} // namespace yul
} // namespace mlir

#endif // MLIR_DIALECT_YUL_TRANSFORMS_PASSES_H

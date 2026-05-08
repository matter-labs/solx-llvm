//===- YulToStandard.h - Yul to Standard dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Yul-to-standard conversion pass and its EVM-specific
// lowering patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_YULTOSTANDARD_YULTOSTANDARD_H
#define MLIR_CONVERSION_YULTOSTANDARD_YULTOSTANDARD_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTYULTOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"

namespace evm {

/// Adds the conversion patterns of yul ops in the sol dialect.
void populateYulPats(RewritePatternSet &pats);

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_YULTOSTANDARD_YULTOSTANDARD_H

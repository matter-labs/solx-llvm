//===- YulToStandard.h - Yul to Standard dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EVM-specific lowering patterns for the Yul dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOSTANDARD_YULTOSTANDARD_H
#define MLIR_CONVERSION_SOLTOSTANDARD_YULTOSTANDARD_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace evm {

/// Adds the conversion patterns of yul ops in the sol dialect.
void populateYulPats(RewritePatternSet &pats);

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_YULTOSTANDARD_H

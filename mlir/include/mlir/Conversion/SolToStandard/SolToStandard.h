//===- SolToStandard.h - Sol to Standard dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sol dialect to standard dialects conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOSTANDARD_SOLTOSTANDARD_H
#define MLIR_CONVERSION_SOLTOSTANDARD_SOLTOSTANDARD_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class Pass;
class TypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTSOLTOSTANDARDPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_SOLTOSTANDARD_H

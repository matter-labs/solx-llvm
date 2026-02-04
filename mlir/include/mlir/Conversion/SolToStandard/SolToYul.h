//===- SolToYul.h - Sol to Yul dialect conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EVM-specific lowering patterns for the Sol dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOSTANDARD_SOLTOYUL_H
#define MLIR_CONVERSION_SOLTOSTANDARD_SOLTOYUL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace evm {

class SolTypeConverter : public TypeConverter {
public:
  SolTypeConverter();
};

/// Adds the conversion patterns of unchecked arithmetic ops in the sol dialect.
void populateArithPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of crypto ops in the sol dialect.
void populateCryptoPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of checked arithmetic ops in the sol dialect.
void populateCheckedArithPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of sol dialect ops dealing with stack, memory
/// and storage allocations.
///
/// Stack access will have 32-byte alignment while all other access will have
/// 1-byte alignment.
void populateMemPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.emit.
void populateEmitPat(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.require.
void populateRequirePat(RewritePatternSet &pats);

/// Adds the conversion pattern of sol.this.
void populateAddrPat(RewritePatternSet &pats);

/// Adds the conversion patterns of abi ops.
void populateAbiPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.ext_call.
void populateExtCallPat(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of control flow ops in the sol dialect.
void populateControlFlowPats(RewritePatternSet &pats);

/// Adds the conversion patterns of func and related ops in the sol dialect.
void populateFuncPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.contract.
void populateContractPat(RewritePatternSet &pats);

void populateStage1Pats(RewritePatternSet &pats, TypeConverter &tyConv);

void populateStage2Pats(RewritePatternSet &pats);

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_SOLTOYUL_H

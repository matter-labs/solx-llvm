//===- SolToYul.h - Sol to Yul dialect conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Sol-to-Yul conversion pass and its EVM-specific
// lowering patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOYUL_SOLTOYUL_H
#define MLIR_CONVERSION_SOLTOYUL_SOLTOYUL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTSOLTOYULPASS
#include "mlir/Conversion/Passes.h.inc"

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

/// Adds the conversion patterns for address/block/tx/msg/gas-related ops.
void populateAddrPat(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of abi ops.
void populateAbiPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.ext_call.
void populateExtCallPat(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of control flow ops in the sol dialect.
void populateControlFlowPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion patterns of sol.call and sol.return.
void populateFuncBoundaryPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.func.
void populateFuncOpPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds late Sol-to-Yul patterns that require legalized signatures and
/// contract runtime/creation placement.
void populateLateSolToYulPats(RewritePatternSet &pats, TypeConverter &tyConv);

/// Adds the conversion pattern of sol.contract.
void populateContractPat(RewritePatternSet &pats);

/// Adds patterns for sol.inline_asm and related ops.
void populateInlineAsmPats(RewritePatternSet &pats, TypeConverter &tyConv);

void populateStage1Pats(RewritePatternSet &pats, TypeConverter &tyConv);

void populateStage2Pats(RewritePatternSet &pats, TypeConverter &tyConv);

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOYUL_SOLTOYUL_H

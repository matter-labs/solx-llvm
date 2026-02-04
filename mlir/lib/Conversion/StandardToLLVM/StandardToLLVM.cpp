//===- StandardToLLVM.cpp - Sol to Standard dialect lowering ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to convert standard dialects to llvm dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToLLVM/StandardToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"

using namespace mlir;

namespace {

struct ConvertStandardToLLVM
    : public PassWrapper<ConvertStandardToLLVM, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStandardToLLVM)

  /// LLVM target triple.
  StringRef triple;

  /// Bitwidth of index type.
  unsigned indexBitwidth;

  /// LLVM target datalayout.
  StringRef dataLayout;

  explicit ConvertStandardToLLVM(StringRef triple, unsigned indexBitwidth,
                                 StringRef dataLayout)
      : triple(triple), indexBitwidth(indexBitwidth), dataLayout(dataLayout) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    LLVMConversionTarget tgt(getContext());
    tgt.addLegalOp<ModuleOp>();

    // Create the llvm type converter.
    LowerToLLVMOptions opts(&getContext());
    opts.overrideIndexBitwidth(indexBitwidth);
    opts.dataLayout = llvm::DataLayout(dataLayout);
    LLVMTypeConverter tyConv(&getContext(), opts);

    // Set the llvm target triple and data-layout for all the module ops (A
    // child module, for instance, represents the runtime context in the ir for
    // evm).
    ModuleOp mod = getOperation();
    mod.walk([&](ModuleOp mod) {
      mod->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                   StringAttr::get(&getContext(), triple));
      mod->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                   StringAttr::get(&getContext(), dataLayout));
    });

    // Populate patterns.
    RewritePatternSet pats(&getContext());
    populateFuncToLLVMConversionPatterns(tyConv, pats);
    populateSCFToControlFlowConversionPatterns(pats);
    cf::populateControlFlowToLLVMConversionPatterns(tyConv, pats);
    mlir::arith::populateArithToLLVMConversionPatterns(tyConv, pats);

    // Run the conversion.
    if (failed(applyFullConversion(mod, tgt, std::move(pats))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
sol::createConvertStandardToLLVMPass(StringRef triple, unsigned indexBitwidth,
                                     StringRef dataLayout) {
  return std::make_unique<ConvertStandardToLLVM>(triple, indexBitwidth,
                                                 dataLayout);
}

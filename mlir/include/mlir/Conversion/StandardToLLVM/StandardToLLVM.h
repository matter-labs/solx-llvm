//===- StandardToLLVM.h - Standard to LLVM dialect conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the standard dialects to LLVM dialect conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {

class Pass;

namespace sol {

/// Creates a pass to convert standard dialects to llvm dialect.
std::unique_ptr<Pass> createConvertStandardToLLVMPass(StringRef triple,
                                                      unsigned indexBitwidth,
                                                      StringRef dataLayout);

} // namespace sol
} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_STANDARDTOLLVM_H

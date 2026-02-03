//===- YulOps.cpp - MLIR operations for Yul implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Yul operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"

using namespace mlir;
using namespace mlir::yul;

#define GET_OP_CLASSES
#include "mlir/Dialect/Yul/YulOps.cpp.inc"

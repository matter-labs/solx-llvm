//===- YulDialect.cpp - MLIR Dialect for Yul implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Yul dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Yul/Yul.h"

#include "mlir/Dialect/Yul/YulOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::yul;

void YulDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Yul/YulOps.cpp.inc"
      >();
}

Operation *YulDialect::materializeConstant(OpBuilder &builder, Attribute val,
                                           Type type, Location loc) {
  return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(val));
}

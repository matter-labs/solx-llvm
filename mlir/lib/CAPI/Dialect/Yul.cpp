//===- Yul.cpp - C API for Yul dialect ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Yul.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Yul/Yul.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Yul, yul, mlir::yul::YulDialect)

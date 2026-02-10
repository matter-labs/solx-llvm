//===-- mlir-c/Dialect/Sol.h - C API for Sol dialect --------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_SOL_H
#define MLIR_C_DIALECT_SOL_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sol, sol);

/// Lowers llvm.setimmutable ops using the provided immutable offset map.
MLIR_CAPI_EXPORTED void mlirEvmLowerSetImmutables(MlirModule mod,
                                                  const char **immIDs,
                                                  const uint64_t *immOffsets,
                                                  uint64_t immCount);

/// Removes all llvm.setimmutable ops.
MLIR_CAPI_EXPORTED void mlirEvmRemoveSetImmutables(MlirModule mod);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/Sol/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_SOL_H

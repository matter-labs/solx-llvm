//===-- Passes.td - Yul pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_YUL_TRANSFORMS_PASSES
#define MLIR_DIALECT_YUL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FuseFreePtrPass : Pass<"yul-fuse-free-ptr"> {
  let summary = "Fuse free pointer operations in the Yul dialect";
}

#endif

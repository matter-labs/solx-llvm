//===-- Passes.td - Sol pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_TRANSFORMS_PASSES
#define MLIR_DIALECT_SOL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ModifierOpLoweringPass : Pass<"sol-lower-modifier", "ModuleOp"> {
  let summary = "Lower modifier-related ops in the Sol dialect";
}

def LoopInvariantCodeMotionPass : Pass<"sol-licm"> {
  let summary = "Loop invariant code motion for the Sol dialect";
}

#endif

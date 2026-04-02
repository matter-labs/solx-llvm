//===- SolImmutables.cpp - Lower Solidity immutables ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering utilities for Solidity immutables.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Sol/Transforms/SolImmutables.h"
#include "mlir/Conversion/SolToStandard/EVMUtil.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

void evm::lowerSetImmutables(ModuleOp mod,
                             llvm::StringMap<SmallVector<uint64_t>> immMap) {
  mod.walk([&](LLVM::SetImmutableOp immOp) {
    auto it = immMap.find(immOp.getName());
    assert(it != immMap.end());
    for (uint64_t offset : it->second) {
      Location loc = immOp.getLoc();
      OpBuilder b(immOp);
      evm::Builder evmB(mod, b, loc);

      auto i256Ty = IntegerType::get(b.getContext(), 256);
      auto offsetConst = b.create<LLVM::ConstantOp>(
          loc, i256Ty, IntegerAttr::get(i256Ty, offset));
      Value addr = evmB.genHeapPtr(
          b.create<LLVM::AddOp>(loc, immOp.getAddr(), offsetConst));
      b.create<LLVM::StoreOp>(loc, immOp.getVal(), addr,
                              evm::getAlignment(addr));
      immOp.erase();
    }
  });
}

void evm::removeSetImmutables(ModuleOp mod) {
  mod.walk([&](LLVM::SetImmutableOp immOp) { immOp->erase(); });
}

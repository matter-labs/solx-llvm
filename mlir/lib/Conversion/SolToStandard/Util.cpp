//===- Util.cpp - MLIR Solidity utilities ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR utility classes for Solidity codegen.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SolToStandard/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

using namespace mlir;
using namespace mlir::solgen;

Value BuilderExt::genIntCast(unsigned width, bool isSigned, Value val,
                             std::optional<Location> locArg) {
  auto srcType = cast<IntegerType>(val.getType());
  assert(srcType.isSignless());
  auto dstSignlessType = b.getIntegerType(width);

  Location loc = locArg ? *locArg : defLoc;

  if (srcType == dstSignlessType)
    return val;
  if (srcType.getWidth() > width)
    return b.create<arith::TruncIOp>(loc, dstSignlessType, val);
  if (isSigned)
    return b.create<arith::ExtSIOp>(loc, dstSignlessType, val);
  return b.create<arith::ExtUIOp>(loc, dstSignlessType, val);
}

Value BuilderExt::genIntCastWithBoolCleanup(unsigned width, bool isSigned,
                                            Value val,
                                            std::optional<Location> locArg,
                                            bool maskBoolAsStorageByte) {
  auto srcType = cast<IntegerType>(val.getType());
  if (width == 1 && srcType.getWidth() > 1) {
    Location loc = locArg ? *locArg : defLoc;
    if (maskBoolAsStorageByte) {
      Value lowBitsMask = b.create<arith::ConstantOp>(
          loc, b.getIntegerAttr(srcType,
                                APInt::getLowBitsSet(srcType.getWidth(), 8)));
      val = b.create<arith::AndIOp>(loc, val, lowBitsMask);
    }
    Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(srcType, 0));
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, val, zero);
  }
  return genIntCast(width, isSigned, val, locArg);
}

sol::FuncOp BuilderExt::getOrInsertFuncOp(StringRef name, FunctionType fnTy,
                                          LLVM::Linkage linkage, ModuleOp mod,
                                          std::vector<NamedAttribute> attrs) {
  if (auto found = mod.lookupSymbol<sol::FuncOp>(name))
    return found;

  // Set insertion point to the ModuleOp's body.
  auto *ctx = mod.getContext();
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(mod.getBody());

  // Add the linkage attribute.
  attrs.emplace_back(StringAttr::get(ctx, "llvm.linkage"),
                     LLVM::LinkageAttr::get(ctx, linkage));

  auto fn = b.create<sol::FuncOp>(mod.getLoc(), name, fnTy, attrs);
  fn.setPrivate();
  return fn;
}

void BuilderExt::createCallToUnreachableWrapper(
    ModuleOp mod, std::optional<Location> locArg) {
  auto fnTy = FunctionType::get(mod.getContext(), {}, {});
  sol::FuncOp fn =
      getOrInsertFuncOp(".unreachable", fnTy, LLVM::Linkage::Private, mod);
  Location loc = locArg ? *locArg : defLoc;
  b.create<sol::CallOp>(
      loc, FlatSymbolRefAttr::get(mod.getContext(), ".unreachable"),
      TypeRange{}, ValueRange{});

  // Define the wrapper if we haven't already.
  if (fn.getBody().empty()) {
    Block *blk = b.createBlock(&fn.getBody());
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(blk);
    b.create<LLVM::UnreachableOp>(loc);
  }
}

Value BuilderExt::genLLVMStruct(ValueRange vals,
                                std::optional<Location> locArg) {
  Location loc = locArg ? *locArg : defLoc;

  SmallVector<Type> eltTys;
  for (Value v : vals)
    eltTys.push_back(v.getType());
  auto structTy = LLVM::LLVMStructType::getLiteral(b.getContext(), eltTys);

  Value res = b.create<LLVM::UndefOp>(loc, structTy);
  for (auto [i, v] : llvm::enumerate(vals))
    res = b.create<LLVM::InsertValueOp>(loc, structTy, res, v,
                                        b.getDenseI64ArrayAttr(i));
  return res;
}

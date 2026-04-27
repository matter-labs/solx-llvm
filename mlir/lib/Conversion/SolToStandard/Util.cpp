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

LLVM::GlobalOp BuilderExt::getStringLiteralGlobalOp(llvm::StringRef literal,
                                                    ModuleOp mod) {
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(mod.getBody());
  auto strAttr = b.getStringAttr(literal);
  Type i8Type = b.getIntegerType(8);
  Type arrayType = LLVM::LLVMArrayType::get(i8Type, literal.size());

  // Generate a unique name of a global that doesn't clash
  // with existing symbols.
  mlir::SymbolTable symbolTable(mod);
  unsigned uniquingCounter = 0;
  llvm::SmallString<128> globName = mlir::SymbolTable::generateSymbolName<128>(
      "__data_in_code_",
      [&](llvm::StringRef candidate) {
        // Return true if name is already taken
        return symbolTable.lookup(candidate) != nullptr;
      },
      uniquingCounter);

  // TODO: deduplicate by content hash to avoid emitting separate globals for
  // identical string literals.
  auto globConst = b.create<LLVM::GlobalOp>(
      defLoc, arrayType,
      /*isConstant=*/true, LLVM::Linkage::Private, globName, strAttr,
      /*alignment=*/0,
      /*addrSpace=*/4);
  globConst.setUnnamedAddr(LLVM::UnnamedAddr::Global);

  return globConst;
}

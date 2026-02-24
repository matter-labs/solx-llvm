//===- Util.h - MLIR Solidity utilities -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares MLIR utility classes for Solidity codegen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOSTANDARD_UTIL_H
#define MLIR_CONVERSION_SOLTOSTANDARD_UTIL_H

#include "mlir/Conversion/SolToStandard/EVMConstants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
#include <utility>

namespace mlir {
namespace solgen {

/// Generates the round-up to multiple.
template <unsigned multiple>
unsigned getRoundUpToMultiple(unsigned val) {
  return ((val + (multiple - 1)) / multiple) * multiple;
}

/// Extension of mlir::OpBuilder with APIs helpful for codegen in solidity.
class BuilderExt {
  OpBuilder &b;
  Location defLoc;

public:
  explicit BuilderExt(OpBuilder &b) : b(b), defLoc(b.getUnknownLoc()) {}

  explicit BuilderExt(OpBuilder &b, Location loc) : b(b), defLoc(loc) {}

  Value genBool(bool val, std::optional<Location> locArg = std::nullopt) {
    IntegerType ty = b.getIntegerType(1);
    auto op = b.create<arith::ConstantOp>(locArg ? *locArg : defLoc,
                                          b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  Value genConst(llvm::APInt const &val, unsigned width,
                 std::optional<Location> locArg = std::nullopt) {
    IntegerType ty = b.getIntegerType(width);
    auto op = b.create<arith::ConstantOp>(locArg ? *locArg : defLoc,
                                          b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  Value genConst(llvm::APInt const &val,
                 std::optional<Location> locArg = std::nullopt) {
    IntegerType ty = b.getIntegerType(val.getBitWidth());
    auto op = b.create<arith::ConstantOp>(locArg ? *locArg : defLoc,
                                          b.getIntegerAttr(ty, val));
    return op.getResult();
  }

  Value genConst(int64_t val, unsigned width = 64,
                 std::optional<Location> locArg = std::nullopt) {
    return genConst(llvm::APInt(width, val, /*isSigned=*/true), width, locArg);
  }

  Value genConst(std::string const &val, unsigned width,
                 std::optional<Location> locArg = std::nullopt) {
    uint8_t radix = 10;
    llvm::StringRef intStr = val;
    if (intStr.consume_front("0x")) {
      radix = 16;
    }

    return genConst(llvm::APInt(width, intStr, radix), width, locArg);
  }

  Value genI256Const(llvm::APInt const &val,
                     std::optional<Location> locArg = std::nullopt) {

    if (val.getBitWidth() != 256) {
      assert(val.getBitWidth() < 256);
      return genConst(val.zext(256), 256, locArg);
    }
    return genConst(val, 256, locArg);
  }

  Value genI256Const(std::string const &val,
                     std::optional<Location> locArg = std::nullopt) {
    return genConst(val, 256, locArg);
  }

  Value genI256Const(int64_t val,
                     std::optional<Location> locArg = std::nullopt) {
    return genConst(val, 256, locArg);
  }

  /// Generates an i256 constant with the selector in the high 32 bits.
  Value genI256Selector(StringRef signature,
                        std::optional<Location> locArg = std::nullopt) {
    Location loc = locArg ? *locArg : defLoc;

    return genI256Const(evm::selectorFromSignatureU256(signature), loc);
  }

  /// Generates an arith dialect cast op (if required) (as per the desired width
  /// and signedness) of the signless typed value.
  Value genIntCast(unsigned width, bool isSigned, Value val,
                   std::optional<Location> locArg = std::nullopt);

  /// Same as genIntCast, except bool types use 'x != 0' cleanup semantics
  // (same as Yul) instead of truncation. If maskBoolAsStorageByte is true,
  // bool cleanup first masks to low 8 bits (storage bool semantics).
  Value genIntCastWithBoolCleanup(unsigned width, bool isSigned, Value val,
                                  std::optional<Location> locArg = std::nullopt,
                                  bool maskBoolAsStorageByte = false);

  Value genCastToIdx(Value val, std::optional<Location> locArg = std::nullopt) {
    assert(val.getType() == b.getIntegerType(256));
    return b.create<arith::IndexCastUIOp>(locArg ? *locArg : defLoc,
                                          b.getIndexType(), val);
  }

  Value genCastToI256(Value val,
                      std::optional<Location> locArg = std::nullopt) {
    // TODO: Support other source types.
    assert(val.getType() == b.getIndexType());
    return b.create<arith::IndexCastUIOp>(locArg ? *locArg : defLoc,
                                          b.getIntegerType(256), val);
  }

  Value genIdxConst(int64_t val,
                    std::optional<Location> locArg = std::nullopt) {
    return b.create<arith::ConstantOp>(locArg ? *locArg : defLoc,
                                       b.getIndexAttr(val));
  }

  /// Generates the round-up to a power-of-2 multiple.
  template <unsigned multiple>
  Value genRoundUpToMultiple(Value val,
                             std::optional<Location> locArg = std::nullopt) {
    static_assert(llvm::isPowerOf2_32(multiple));
    Location loc = locArg ? *locArg : defLoc;
    auto add = b.create<arith::AddIOp>(loc, val, genI256Const(multiple - 1));
    return b.create<arith::AndIOp>(
        loc, add, genI256Const(~(llvm::APInt(/*numBits=*/256, multiple - 1))));
  }

  /// Generates the ceiling of value / multiple (power-of-2).
  template <unsigned multiple>
  mlir::Value
  genCeilDivision(mlir::Value val,
                  std::optional<mlir::Location> locArg = std::nullopt) {
    static_assert(llvm::isPowerOf2_32(multiple));
    mlir::Location loc = locArg ? *locArg : defLoc;
    auto add =
        b.create<mlir::arith::AddIOp>(loc, val, genI256Const(multiple - 1));
    return b.create<mlir::arith::DivUIOp>(loc, add, genI256Const(multiple));
  }

  /// Returns an existing or a new (if not found) FuncOp in the ModuleOp `mod`.
  sol::FuncOp getOrInsertFuncOp(StringRef name, FunctionType fnTy,
                                LLVM::Linkage linkage, ModuleOp mod,
                                std::vector<NamedAttribute> attrs = {});

  /// Creates a private constant LLVM global in address space 4 (code) holding
  /// the given string literal bytes, with a unique \c __data_in_code_* name.
  mlir::LLVM::GlobalOp getStringLiteralGlobalOp(llvm::StringRef literal,
                                                mlir::ModuleOp mod);

  /// Creates a call to a wrapper function of the LLVM::UnreachableOp. This is a
  /// hack to create a non-terminator unreachable op
  void
  createCallToUnreachableWrapper(ModuleOp mod,
                                 std::optional<Location> locArg = std::nullopt);

  /// Generates an LLVM literal struct from the given values.
  Value genLLVMStruct(ValueRange vals,
                      std::optional<Location> locArg = std::nullopt);
};

} // namespace solgen
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_UTIL_H

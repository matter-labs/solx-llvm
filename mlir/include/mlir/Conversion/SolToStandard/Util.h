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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"
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
#include "llvm/ADT/STLExtras.h"
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

  Value genConst(llvm::APInt const &val,
                 std::optional<Location> locArg = std::nullopt) {
    assert(val.getBitWidth() == 256);
    IntegerType ty = b.getIntegerType(256);
    auto op = b.create<yul::ConstantOp>(locArg ? *locArg : defLoc,
                                        b.getIntegerAttr(ty, val));
    return op.getResult();
  }

public:
  explicit BuilderExt(OpBuilder &b) : b(b), defLoc(b.getUnknownLoc()) {}

  explicit BuilderExt(OpBuilder &b, Location loc) : b(b), defLoc(loc) {}

  Value genBool(bool val, std::optional<Location> locArg = std::nullopt) {
    return genConst(llvm::APInt(256, val), locArg);
  }

  Value genI256Const(llvm::APInt const &val,
                     std::optional<Location> locArg = std::nullopt) {
    return genConst(val, locArg);
  }

  Value genI256Const(int64_t val,
                     std::optional<Location> locArg = std::nullopt) {
    return genConst(llvm::APInt(256, val, /*isSigned=*/true), locArg);
  }

  /// Generates an i256 constant with the selector in the high 32 bits.
  Value genI256Selector(StringRef signature,
                        std::optional<Location> locArg = std::nullopt) {
    Location loc = locArg ? *locArg : defLoc;

    return genI256Const(evm::selectorFromSignatureU256(signature), loc);
  }

  Value genCmp(yul::CmpPredicate predicate, Value lhs, Value rhs,
               std::optional<Location> locArg = std::nullopt) {
    return b.create<yul::CmpOp>(locArg ? *locArg : defLoc,
                                b.getIntegerType(256), predicate, lhs, rhs);
  }

  yul::IfOp createIf(TypeRange resultTypes, Value cond,
                     bool withElseRegion = false,
                     std::optional<Location> locArg = std::nullopt) {
    Location loc = locArg ? *locArg : defLoc;
    auto ifOp = b.create<yul::IfOp>(loc, resultTypes, cond);
    ifOp.getThenRegion().emplaceBlock();
    if (withElseRegion || !resultTypes.empty())
      ifOp.getElseRegion().emplaceBlock();

    // Resultful yul.if regions must yield values chosen by the caller.
    if (!resultTypes.empty())
      return ifOp;

    // Resultless yul.if regions need an empty terminator; pre-create it so
    // callers can insert the body at the start, matching scf.if builder style.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&ifOp.getThenRegion().front());
    b.create<yul::YieldOp>(loc);
    if (!ifOp.getElseRegion().empty()) {
      b.setInsertionPointToEnd(&ifOp.getElseRegion().front());
      b.create<yul::YieldOp>(loc);
    }
    return ifOp;
  }

  yul::IfOp createIf(Type resultType, Value cond, bool withElseRegion = false,
                     std::optional<Location> locArg = std::nullopt) {
    return createIf(TypeRange{resultType}, cond, withElseRegion, locArg);
  }

  yul::IfOp createIf(Value cond, bool withElseRegion = false,
                     std::optional<Location> locArg = std::nullopt) {
    return createIf(TypeRange{}, cond, withElseRegion, locArg);
  }

  /// Creates a counted yul.for with optional loop-carried values.
  ///
  /// Pseudo MLIR shape:
  ///
  ///   %iv_final, %arg0_final, ... = yul.for (%iv = %lower,
  ///                                           %arg0 = %init0, ...)
  ///     cond(%iv, %arg0, ...) {
  ///       %keep_going = yul.cmp ult, %iv, %upper
  ///       yul.condition %keep_going, %iv, %arg0, ...
  ///     }
  ///     body(%iv, %arg0, ...) {
  ///       %next_arg0, ... = bodyBuilder(%iv, [%arg0, ...])
  ///       yul.yield %iv, %next_arg0, ...
  ///     }
  ///     step(%iv, %arg0, ...) {
  ///       %next_iv = yul.add %iv, %step
  ///       yul.yield %next_iv, %arg0, ...
  ///     }
  ///
  /// The returned yul.for exposes the final induction variable as result 0.
  /// Caller-provided loop-carried values follow it, so initArgs[i] maps to
  /// forOp.getResult(i + 1). The bodyBuilder receives the induction variable
  /// separately from ValueRange{arg0, ...} and returns only the next values for
  /// those caller-provided loop-carried arguments.
  template <typename BodyBuilder>
  yul::ForOp createCountedLoop(Value lowerBound, Value upperBound, Value step,
                               ValueRange initArgs, BodyBuilder &&bodyBuilder,
                               bool fullUnroll = false,
                               std::optional<Location> locArg = std::nullopt) {
    Location loc = locArg ? *locArg : defLoc;

    SmallVector<Value> inits{lowerBound};
    llvm::append_range(inits, initArgs);
    TypeRange loopTypes(inits);
    SmallVector<Location> argLocs(inits.size(), loc);

    auto forOp = b.create<yul::ForOp>(loc, loopTypes, inits);
    if (fullUnroll)
      forOp.setFullUnrollAttr(b.getUnitAttr());

    Block *condBlock = &forOp.getCond().emplaceBlock();
    forOp.getCond().addArguments(loopTypes, argLocs);
    b.setInsertionPointToStart(condBlock);
    Value iv = condBlock->getArgument(0);
    Value cond = genCmp(yul::CmpPredicate::ult, iv, upperBound, loc);
    b.create<yul::ConditionOp>(loc, cond, condBlock->getArguments());

    Block *bodyBlock = &forOp.getBody().emplaceBlock();
    forOp.getBody().addArguments(loopTypes, argLocs);
    b.setInsertionPointToStart(bodyBlock);
    ValueRange bodyArgs(bodyBlock->getArguments());
    SmallVector<Value> nextArgs =
        bodyBuilder(b, loc, bodyArgs.front(), bodyArgs.drop_front());

    SmallVector<Value> bodyYieldArgs{bodyArgs.front()};
    llvm::append_range(bodyYieldArgs, nextArgs);
    b.create<yul::YieldOp>(loc, bodyYieldArgs);

    Block *stepBlock = &forOp.getStep().emplaceBlock();
    forOp.getStep().addArguments(loopTypes, argLocs);
    b.setInsertionPointToStart(stepBlock);
    ValueRange stepArgs(stepBlock->getArguments());
    Value nextIV = b.create<yul::AddOp>(loc, stepArgs.front(), step);
    SmallVector<Value> stepYieldArgs{nextIV};
    llvm::append_range(stepYieldArgs, stepArgs.drop_front());
    b.create<yul::YieldOp>(loc, stepYieldArgs);

    b.setInsertionPointAfter(forOp);
    return forOp;
  }

  /// Generates the round-up to a power-of-2 multiple.
  template <unsigned multiple>
  Value genRoundUpToMultiple(Value val,
                             std::optional<Location> locArg = std::nullopt) {
    static_assert(llvm::isPowerOf2_32(multiple));
    Location loc = locArg ? *locArg : defLoc;
    auto add = b.create<yul::AddOp>(loc, val, genI256Const(multiple - 1));
    return b.create<yul::AndOp>(
        loc, add, genI256Const(~(llvm::APInt(/*numBits=*/256, multiple - 1))));
  }

  /// Generates the ceiling of value / multiple (power-of-2).
  template <unsigned multiple>
  mlir::Value
  genCeilDivision(mlir::Value val,
                  std::optional<mlir::Location> locArg = std::nullopt) {
    static_assert(llvm::isPowerOf2_32(multiple));
    mlir::Location loc = locArg ? *locArg : defLoc;
    auto add = b.create<mlir::yul::AddOp>(loc, val, genI256Const(multiple - 1));
    return b.create<mlir::yul::ArithDivOp>(loc, add, genI256Const(multiple));
  }

  /// Creates a private constant LLVM global in address space 4 (code) holding
  /// the given string literal bytes, with a unique \c __data_in_code_* name.
  mlir::LLVM::GlobalOp getStringLiteralGlobalOp(llvm::StringRef literal,
                                                mlir::ModuleOp mod);

  /// Generates an LLVM literal struct from the given values.
  Value genLLVMStruct(ValueRange vals,
                      std::optional<Location> locArg = std::nullopt);
};

} // namespace solgen
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_UTIL_H

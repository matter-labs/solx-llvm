//===- SolToYul.cpp - Sol to Yul dialect conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EVM specific lowering patterns for the Sol dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SolToStandard/SolToYul.h"
#include "mlir/Conversion/SolToStandard/EVMConstants.h"
#include "mlir/Conversion/SolToStandard/EVMUtil.h"
#include "mlir/Conversion/SolToStandard/Util.h"
#include "mlir/Conversion/SolToStandard/YulToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {

struct ConstantOpLowering : public OpRewritePattern<sol::ConstantOp> {
  using OpRewritePattern<sol::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ConstantOp op,
                                PatternRewriter &r) const override {
    auto signlessTy =
        r.getIntegerType(cast<IntegerType>(op.getType()).getWidth());
    auto attr = cast<IntegerAttr>(op.getValue());
    r.replaceOpWithNewOp<arith::ConstantOp>(
        op, signlessTy, r.getIntegerAttr(signlessTy, attr.getValue()));
    return success();
  }
};

struct FuncConstantOpLowering : public OpRewritePattern<sol::FuncConstantOp> {
  using OpRewritePattern<sol::FuncConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::FuncConstantOp op,
                                PatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());
    auto fn =
        SymbolTable::lookupNearestSymbolFrom<sol::FuncOp>(op, op.getSymAttr());
    std::optional<uint64_t> id = fn.getId();
    assert(id);
    r.replaceOp(op, bExt.genI256Const(*id));
    return success();
  }
};

struct DefaultFuncConstantOpLowering
    : public OpRewritePattern<sol::DefaultFuncConstantOp> {
  using OpRewritePattern<sol::DefaultFuncConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::DefaultFuncConstantOp op,
                                PatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());
    r.replaceOp(op, bExt.genI256Const(0));
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<sol::CastOp> {
  using OpConversionPattern<sol::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto inpTy = dyn_cast<IntegerType>(op.getInp().getType());
    if (!inpTy) {
      assert(isa<sol::EnumType>(op.getInp().getType()));
      inpTy = r.getIntegerType(256);
    }
    auto inpTyWidth = inpTy.getWidth();
    auto outTyWidth = cast<IntegerType>(op.getType()).getWidth();

    if (inpTyWidth == outTyWidth) {
      r.replaceOp(op, adaptor.getInp());
      return success();
    }

    IntegerType signlessOutTy = r.getIntegerType(outTyWidth);

    if (inpTyWidth > outTyWidth) {
      r.replaceOpWithNewOp<arith::TruncIOp>(op, signlessOutTy,
                                            adaptor.getInp());
      return success();
    }
    if (inpTy.isSigned())
      r.replaceOpWithNewOp<arith::ExtSIOp>(op, signlessOutTy, adaptor.getInp());
    else
      r.replaceOpWithNewOp<arith::ExtUIOp>(op, signlessOutTy, adaptor.getInp());
    return success();
  }
};

struct AddressCastOpLowering : public OpConversionPattern<sol::AddressCastOp> {
  using OpConversionPattern<sol::AddressCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::AddressCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    Type inpTy = op.getInp().getType();
    Type outTy = op.getType();

    if (isa<sol::BytesType>(inpTy) || isa<sol::BytesType>(outTy)) {
      // bytes20 -> address
      if (auto inpBytesTy = dyn_cast<sol::BytesType>(inpTy)) {
        assert(inpBytesTy.getSize() == 20 &&
               "AddressCastOp only supports bytes20");
        assert(isa<sol::AddressType>(outTy));
        r.replaceOpWithNewOp<arith::ShRUIOp>(op, adaptor.getInp(),
                                             bExt.genI256Const(96));
        return success();
      }

      // address -> bytes20
      auto outBytesTy = cast<sol::BytesType>(outTy);
      assert(outBytesTy.getSize() == 20 &&
             "AddressCastOp only supports bytes20");
      assert(isa<sol::AddressType>(inpTy));
      r.replaceOpWithNewOp<arith::ShLIOp>(op, adaptor.getInp(),
                                          bExt.genI256Const(96));
      return success();
    }

    if (isa<IntegerType>(inpTy) || isa<IntegerType>(outTy)) {
      // uint160 -> address
      if (isa<IntegerType>(inpTy)) {
        auto inpIntTy = cast<IntegerType>(inpTy);
        assert(inpIntTy.getWidth() == 160 && !inpIntTy.isSigned() &&
               "AddressCastOp only supports uint160 -> address");
        assert(isa<sol::AddressType>(outTy));
        // Zero-extend uint160 to 256 bits for the address representation.
        Value zext = bExt.genIntCast(/*width=*/256, /*isSigned=*/false,
                                     adaptor.getInp(), loc);
        r.replaceOp(op, zext);
        return success();
      }

      // address -> uint160
      auto outIntTy = cast<IntegerType>(outTy);
      assert(isa<sol::AddressType>(inpTy) && outIntTy.getWidth() == 160 &&
             !outIntTy.isSigned() &&
             "AddressCastOp only supports address -> uint160");
      // Truncate the 256-bit address representation to uint160.
      Value trunc = bExt.genIntCast(/*width=*/160, /*isSigned=*/false,
                                    adaptor.getInp(), loc);
      r.replaceOp(op, trunc);
      return success();
    }

    if (isa<sol::AddressType>(inpTy) && isa<sol::AddressType>(outTy))
      // address -> address (payable <-> non-payable)
      assert(
          cast<sol::AddressType>(inpTy).getPayable() !=
              cast<sol::AddressType>(outTy).getPayable() &&
          "AddressCastOp only supports payable <-> non-payable address casts");
    else if (!(isa<sol::AddressType>(inpTy) && isa<sol::ContractType>(outTy)) &&
             !(isa<sol::ContractType>(inpTy) && isa<sol::AddressType>(outTy)))
      // If not an address <-> contract cast, issue an error.
      llvm_unreachable("Unexpected types for AddressCastOp");

    // TODO: This is probably not needed, but it is here for Yul parity.
    Value mask = bExt.genI256Const(APInt::getLowBitsSet(256, 160));
    r.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getInp(), mask);
    return success();
  }
};

struct ContractCastOpLowering
    : public OpConversionPattern<sol::ContractCastOp> {
  using OpConversionPattern<sol::ContractCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ContractCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    // TODO: This is probably not needed, but it is here for Yul parity.
    Value mask = bExt.genI256Const(APInt::getLowBitsSet(256, 160));
    r.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getInp(), mask);
    return success();
  }
};

struct EnumCastOpLowering : public OpConversionPattern<sol::EnumCastOp> {
  using OpConversionPattern<sol::EnumCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EnumCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto enumTy = dyn_cast<sol::EnumType>(op.getType());
    auto panicCond = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, adaptor.getInp(),
        bExt.genI256Const(enumTy.getMax()));
    evmB.genPanic(mlir::evm::PanicCode::EnumConversionError, panicCond, loc);
    r.replaceOp(op, bExt.genIntCast(256, /*isSigned=*/false, adaptor.getInp()));
    return success();
  }
};

struct BytesCastOpLowering : public OpConversionPattern<sol::BytesCastOp> {
  using OpConversionPattern<sol::BytesCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BytesCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    Type inpTy = op.getInp().getType();
    Type outTy = op.getType();

    // Bytes to bytes
    if (auto inpBytesTy = dyn_cast<sol::BytesType>(inpTy)) {
      if (auto outBytesTy = dyn_cast<sol::BytesType>(outTy)) {
        unsigned keepBytes = inpBytesTy.getSize() < outBytesTy.getSize()
                                 ? inpBytesTy.getSize()
                                 : outBytesTy.getSize();
        auto shiftAmt = bExt.genI256Const(256 - (8 * keepBytes));
        auto shr = r.create<arith::ShRUIOp>(loc, adaptor.getInp(), shiftAmt);
        r.replaceOpWithNewOp<arith::ShLIOp>(op, shr, shiftAmt);
        return success();
      }

      // Bytes to int
      auto outIntTy = cast<IntegerType>(outTy);
      auto shiftAmt = bExt.genI256Const(256 - (8 * inpBytesTy.getSize()));
      auto shr = r.create<arith::ShRUIOp>(loc, adaptor.getInp(), shiftAmt);
      auto repl = bExt.genIntCast(outIntTy.getWidth(), /*isSigned=*/false, shr);
      r.replaceOp(op, repl);
      return success();
    }

    // Int to bytes
    assert(isa<IntegerType>(inpTy));
    auto outBytesTy = cast<sol::BytesType>(outTy);
    Value inpAsI256 =
        bExt.genIntCast(/*width=*/256, /*isSigned=*/false, adaptor.getInp());
    auto shiftAmt = bExt.genI256Const(256 - (8 * outBytesTy.getSize()));
    r.replaceOpWithNewOp<arith::ShLIOp>(op, inpAsI256, shiftAmt);

    return success();
  }
};

/// A templatized version of a conversion pattern for lowering add, sub, mul
/// and exp ops.
template <typename SrcOpT, typename DstOpT>
struct ArithBinOpLowering : public OpConversionPattern<SrcOpT> {
  using OpConversionPattern<SrcOpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(SrcOpT op, typename SrcOpT::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<DstOpT>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// A templatized version of a conversion pattern for lowering div and mod ops.
template <typename SolOp, typename ArithSignedOp, typename ArithUnsignedOp>
struct DivOrModOpLowering : public OpConversionPattern<SolOp> {
  using OpConversionPattern<SolOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SolOp op, typename SolOp::Adaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto zero = bExt.genConst(0, ty.getWidth());
    auto rhsEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, zero);
    evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned())
      r.replaceOpWithNewOp<ArithSignedOp>(op, lhs, rhs);
    else
      r.replaceOpWithNewOp<ArithUnsignedOp>(op, lhs, rhs);
    return success();
  }
};

struct ExpOpLowering : public OpConversionPattern<sol::ExpOp> {
  using OpConversionPattern<sol::ExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    // Yul exp op work with i256.
    Value lhs256 = bExt.genIntCast(256, ty.isSigned(), adaptor.getLhs());
    Value rhs256 = bExt.genIntCast(256, ty.isSigned(), adaptor.getRhs());
    Value exp256 = r.create<yul::ExpOp>(loc, lhs256, rhs256);
    // Cast back to the original bitwidth.
    Value exp = bExt.genIntCast(ty.getWidth(), ty.isSigned(), exp256);
    r.replaceOp(op, exp);
    return success();
  }
};

struct CExpOpLowering : public OpConversionPattern<sol::CExpOp> {
  using OpConversionPattern<sol::CExpOp>::OpConversionPattern;

  // The genExpHelper function implements the following YUL helper function.
  //
  //   function checked_exp_helper(power, base, exponent, max) -> power, base {
  //     for { } gt(exponent, 1) {}
  //     {
  //       if gt(base, div(max, base)) { panic_error_0x11() }
  //       if and(exponent, 1)
  //       {
  //         power := mul(power, base)
  //       }
  //       base := mul(base, base)
  //       exponent := shift_right_1_unsigned(exponent)
  //     }
  //  }

  std::pair<Value, Value> genExpHelper(ConversionPatternRewriter &r,
                                       Location loc, Value initPow,
                                       Value initBase, Value initExp,
                                       Value maxPow) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(initBase.getType());
    Value one = bExt.genConst(1, ty.getWidth(), loc);
    SmallVector<Value> whileInitVals{initPow, initBase, initExp};
    TypeRange whileArgTypes(whileInitVals);
    SmallVector<Location> whileBbArgLocs(whileInitVals.size(), loc);

    // Produces power, base
    auto whileOp = r.create<scf::WhileOp>(loc, whileArgTypes, whileInitVals);
    // Condition region
    {
      Block *beforeBlock = &whileOp.getBefore().emplaceBlock();
      whileOp.getBefore().addArguments(whileArgTypes, whileBbArgLocs);
      r.setInsertionPointToStart(beforeBlock);
      Value curPow = beforeBlock->getArgument(0);
      Value curBase = beforeBlock->getArgument(1);
      Value curExp = beforeBlock->getArgument(2);
      Value cmp =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, curExp, one);
      r.create<scf::ConditionOp>(loc, cmp, ValueRange{curPow, curBase, curExp});
    }

    // Body region
    {
      Block *afterBlock = &whileOp.getAfter().emplaceBlock();
      whileOp.getAfter().addArguments(whileArgTypes, whileBbArgLocs);
      r.setInsertionPointToStart(afterBlock);
      Value curPow = afterBlock->getArgument(0);
      Value curBase = afterBlock->getArgument(1);
      Value curExp = afterBlock->getArgument(2);
      {
        auto base256 = bExt.genIntCast(256, /*isSigned=*/false, curBase);
        Value tmpDiv = r.create<yul::DivOp>(loc, maxPow, base256);
        Value panicCond = r.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ugt, base256, tmpDiv);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
      }
      Value lsb = r.create<arith::AndIOp>(loc, curExp, one);
      Value zero = bExt.genConst(0, ty.getWidth(), loc);
      Value expIsOdd =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lsb, zero);
      Value newPow = r.create<arith::SelectOp>(
                          loc, expIsOdd,
                          r.create<arith::MulIOp>(loc, curPow, curBase), curPow)
                         .getResult();
      Value newBase = r.create<arith::MulIOp>(loc, curBase, curBase);
      Value newExp = r.create<arith::ShRUIOp>(loc, curExp, one);
      r.create<scf::YieldOp>(loc, ValueRange{newPow, newBase, newExp});
    }

    return {whileOp.getResult(0), whileOp.getResult(1)};
  }

  // Below is the implementation of unsigned exponentiation based on the
  // following YUL algorithm.
  //
  //   function checked_exp_unsigned(base, exponent, max) -> power {
  //     if iszero(exponent) { power := 1 leave }
  //     if iszero(base) { power := 0 leave }
  //     switch base
  //     case 1 { power := 1 leave }
  //     case 2
  //     {
  //       if gt(exponent, 255) { panic_error_0x11() }
  //       power := exp(2, exponent)
  //       if gt(power, max) { panic_error_0x11() }
  //       leave
  //     }
  //     if or(and(lt(base, 11), lt(exponent, 78)),
  //           and(lt(base, 307), lt(exponent, 32)))
  //     {
  //       power := exp(base, exponent)
  //       if gt(power, max) { panic_error_0x11() }
  //       leave
  //     }
  //     power, base := checked_exp_helper(1, base, exponent, max)
  //     if gt(power, div(max, base)) { panic_error_0x11() }
  //     power := mul(power, base)
  //   }

  Value expUnsigned(sol::CExpOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &r) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value base = adaptor.getLhs();
    Value exp = adaptor.getRhs();

    // The type of the result is the type of 'base'.
    auto ty = cast<IntegerType>(base.getType());
    Value max256 =
        bExt.genI256Const(llvm::APInt::getMaxValue(ty.getWidth()), loc);
    Value zero = bExt.genConst(0, ty.getWidth(), loc);
    Value one = bExt.genConst(1, ty.getWidth(), loc);

    auto expEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exp, zero);
    auto baseEqOne =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, base, one);
    auto powOneCond = r.create<arith::OrIOp>(loc, expEqZero, baseEqOne);

    // If 1 begin
    auto ifExpEqZero = r.create<scf::IfOp>(loc, ty, powOneCond, true);
    // If 1 then
    r.setInsertionPointToStart(&ifExpEqZero.getThenRegion().front());
    r.create<scf::YieldOp>(loc, one);

    // If 1 else
    r.setInsertionPointToStart(&ifExpEqZero.getElseRegion().front());
    auto baseEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, base, zero);

    // If 2 begin
    auto ifBaseEqZero = r.create<scf::IfOp>(loc, ty, baseEqZero, true);
    // If 2 then
    r.setInsertionPointToStart(&ifBaseEqZero.getThenRegion().front());
    r.create<scf::YieldOp>(loc, zero);

    // If 2 else
    r.setInsertionPointToStart(&ifBaseEqZero.getElseRegion().front());
    Value two = bExt.genConst(2, ty.getWidth(), loc);
    auto baseEqTwo =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, base, two);

    // If 3 begin
    auto ifBaseEqTwo = r.create<scf::IfOp>(loc, ty, baseEqTwo, true);
    // If 3 then
    r.setInsertionPointToStart(&ifBaseEqTwo.getThenRegion().front());
    {
      auto expGt255 = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, exp,
          bExt.genConst(APInt(ty.getWidth(), 255, /*isSigned=*/false), loc));
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, expGt255);

      auto two256 = bExt.genI256Const(2, loc);
      auto expZExt = bExt.genIntCast(256, /*isSigned=*/false, exp, loc);
      auto tmpPow = r.create<yul::ExpOp>(loc, two256, expZExt);
      auto panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                               tmpPow, max256);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);

      // Cast the result back to the original bitwidth.
      auto powCasted =
          bExt.genIntCast(ty.getWidth(), /*isSigned=*/false, tmpPow, loc);
      r.create<scf::YieldOp>(loc, powCasted);
    }
    // If 3 else
    r.setInsertionPointToStart(&ifBaseEqTwo.getElseRegion().front());
    auto baseLT11 =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, base,
                                bExt.genConst(11, ty.getWidth(), loc));
    Value baseLT307 =
        (ty.getWidth() > 8)
            ? r.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::ult, base,
                  bExt.genConst(APInt(ty.getWidth(), 307, /*isSigned*/ false),
                                loc))
            : bExt.genBool(true, loc);

    auto expLT78 =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, exp,
                                bExt.genConst(78, ty.getWidth(), loc));
    auto expLT32 =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, exp,
                                bExt.genConst(32, ty.getWidth(), loc));

    auto viaExpBuitinCond = r.create<arith::OrIOp>(
        loc, r.create<arith::AndIOp>(loc, baseLT11, expLT78),
        r.create<arith::AndIOp>(loc, baseLT307, expLT32));
    // If 4 begin
    auto ifViaExpBuitinCond =
        r.create<scf::IfOp>(loc, ty, viaExpBuitinCond, true);
    // If 4 then
    r.setInsertionPointToStart(&ifViaExpBuitinCond.getThenRegion().front());

    {
      auto expZExt = bExt.genIntCast(256, /*isSigned=*/false, exp, loc);
      auto baseZExt = bExt.genIntCast(256, /*isSigned=*/false, base, loc);
      Value tmpPow = r.create<yul::ExpOp>(loc, baseZExt, expZExt);
      Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                tmpPow, max256);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);

      // Cast the result back to the original bitwidth.
      auto powCasted =
          bExt.genIntCast(ty.getWidth(), /*isSigned=*/false, tmpPow, loc);
      r.create<scf::YieldOp>(loc, powCasted);
    }
    // If 4 else
    r.setInsertionPointToStart(&ifViaExpBuitinCond.getElseRegion().front());
    auto [pow2, base2] = genExpHelper(r, loc, one, base, exp, max256);

    {
      r.setInsertionPointToEnd(&ifViaExpBuitinCond.getElseRegion().front());
      auto baseZExt = bExt.genIntCast(256, /*isSigned=*/false, base2, loc);
      Value tmpDiv = r.create<yul::DivOp>(loc, max256, baseZExt);
      auto pow256 = bExt.genIntCast(256, /*isSigned=*/false, pow2, loc);
      Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                pow256, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
      Value res = r.create<arith::MulIOp>(loc, base2, pow2);
      r.create<scf::YieldOp>(loc, res);
    }
    // If 4 (ifViaExpBuitinCond) end
    r.setInsertionPointToEnd(&ifBaseEqTwo.getElseRegion().front());
    r.create<scf::YieldOp>(loc, ifViaExpBuitinCond.getResult(0));
    // If 3 (ifBaseEqTwo) end

    r.setInsertionPointToEnd(&ifBaseEqZero.getElseRegion().front());
    r.create<scf::YieldOp>(loc, ifBaseEqTwo.getResult(0));
    // If 2 (ifBaseEqZero) end

    r.setInsertionPointToEnd(&ifExpEqZero.getElseRegion().front());
    r.create<scf::YieldOp>(loc, ifBaseEqZero.getResult(0));
    // If 1 (ifExpEqZero) end

    r.setInsertionPointAfter(ifExpEqZero);

    return ifExpEqZero.getResult(0);
  }

  // Below is the implementation of signed exponentiation based on the
  // following YUL algorithm.
  //
  //   function checked_exp_signed(base, exponent, min, max) -> power {
  //     switch exponent
  //     case 0 { power := 1 leave }
  //     case 1 { power := base leave }
  //     if iszero(base) { power := 0 leave }
  //
  //     power := 1
  //     switch sgt(base, 0)
  //     case 1 { if gt(base, div(max, base)) { panic_error_0x11() } }
  //     case 0 { if slt(base, sdiv(max, base)) { panic_error_0x11() } }
  //     if and(exponent, 1)
  //     {
  //       power := base
  //     }
  //     base := mul(base, base)
  //     exponent := shift_right_1_unsigned(exponent)
  //
  //     power, base := checked_exp_helper(power, base, exponent, max)
  //
  //     if and(sgt(power, 0), gt(power, div(max, base))) { panic_error_0x11() }
  //     if and(slt(power, 0), slt(power, sdiv(min, base))) { panic_error_0x11()
  //     } power := mul(power, base)
  //  }

  Value expSigned(sol::CExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &r) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value base = adaptor.getLhs();
    Value exp = adaptor.getRhs();

    // The type of the result is the type of 'base'.
    auto ty = cast<IntegerType>(base.getType());
    Value zero = bExt.genConst(0, ty.getWidth(), loc);
    Value one = bExt.genConst(1, ty.getWidth(), loc);
    Value max256 = bExt.genI256Const(
        llvm::APInt::getSignedMaxValue(ty.getWidth()).sext(256), loc);
    Value min256 = bExt.genI256Const(
        llvm::APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);

    auto expEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exp, zero);

    // If 1 begin
    auto ifExpEqZero = r.create<scf::IfOp>(loc, ty, expEqZero, true);
    // If 1 then
    r.setInsertionPointToStart(&ifExpEqZero.getThenRegion().front());
    r.create<scf::YieldOp>(loc, one);

    // If 1 else
    r.setInsertionPointToStart(&ifExpEqZero.getElseRegion().front());
    auto baseEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, base, zero);

    // If 2 begin
    auto ifBaseEqZero = r.create<scf::IfOp>(loc, ty, baseEqZero, true);
    // If 2 then
    r.setInsertionPointToStart(&ifBaseEqZero.getThenRegion().front());
    r.create<scf::YieldOp>(loc, zero);

    // If 2 else
    r.setInsertionPointToStart(&ifBaseEqZero.getElseRegion().front());
    auto expEqOne =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exp, one);

    // If 3 begin
    auto ifExpEqOne = r.create<scf::IfOp>(loc, ty, expEqOne, true);
    // If 3 then
    r.setInsertionPointToStart(&ifExpEqOne.getThenRegion().front());
    r.create<scf::YieldOp>(loc, base);

    // If 3 else
    r.setInsertionPointToStart(&ifExpEqOne.getElseRegion().front());
    auto baseSgtZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, base, zero);

    // If 4 begin
    auto ifBaseSgtZero = r.create<scf::IfOp>(loc, baseSgtZero, true);
    // If 4 then
    r.setInsertionPointToStart(&ifBaseSgtZero.getThenRegion().front());
    {
      auto baseZExt = bExt.genIntCast(256, /*isSigned=*/false, base, loc);
      Value tmpDiv = r.create<yul::DivOp>(loc, max256, baseZExt);
      Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                                baseZExt, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    // If 4 else
    r.setInsertionPointToStart(&ifBaseSgtZero.getElseRegion().front());
    {
      auto baseSExt = bExt.genIntCast(256, /*isSigned=*/true, base, loc);
      Value tmpDiv = r.create<yul::SDivOp>(loc, max256, baseSExt);
      Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                baseSExt, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    // If 4 end
    r.setInsertionPointAfter(ifBaseSgtZero);

    Value lsb = r.create<arith::AndIOp>(loc, exp, one);
    Value expIsOdd =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lsb, zero);
    Value pow2 =
        r.create<arith::SelectOp>(loc, expIsOdd, base, one).getResult();
    Value base2 = r.create<arith::MulIOp>(loc, base, base);
    Value exp2 = r.create<arith::ShRUIOp>(loc, exp, one);
    auto [pow3, base3] = genExpHelper(r, loc, pow2, base2, exp2, max256);

    r.setInsertionPointToEnd(&ifExpEqOne.getElseRegion().front());
    {
      auto baseZExt = bExt.genIntCast(256, /*isSigned=*/false, base3, loc);
      auto powZExt = bExt.genIntCast(256, /*isSigned=*/false, pow3, loc);
      auto powSExt = bExt.genIntCast(256, /*isSigned=*/true, pow3, loc);
      auto zero256 = bExt.genI256Const(0, loc);
      Value div = r.create<yul::DivOp>(loc, max256, baseZExt);
      Value cmp =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, powZExt, div);
      Value cmp2 = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                           powSExt, zero256);
      Value panicCond = r.create<arith::AndIOp>(loc, cmp, cmp2);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    {
      auto baseSExt = bExt.genIntCast(256, /*isSigned=*/true, base3, loc);
      auto powSExt = bExt.genIntCast(256, /*isSigned=*/true, pow3, loc);
      auto zero256 = bExt.genI256Const(0, loc);
      Value div = r.create<yul::SDivOp>(loc, min256, baseSExt);
      Value cmp =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, powSExt, div);
      Value cmp2 = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                           powSExt, zero256);
      Value panicCond = r.create<arith::AndIOp>(loc, cmp, cmp2);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }

    Value pow4 = r.create<arith::MulIOp>(loc, pow3, base3);
    r.create<scf::YieldOp>(loc, pow4);
    // If 3 (ifExpEqOne) end

    r.setInsertionPointToEnd(&ifBaseEqZero.getElseRegion().front());
    r.create<scf::YieldOp>(loc, ifExpEqOne.getResult(0));
    // If 2 (ifBaseEqZero) end

    r.setInsertionPointToEnd(&ifExpEqZero.getElseRegion().front());
    r.create<scf::YieldOp>(loc, ifBaseEqZero.getResult(0));
    // If 1 (ifExpEqZero) end

    r.setInsertionPointAfter(ifExpEqZero);

    return ifExpEqZero.getResult(0);
  }

  LogicalResult matchAndRewrite(sol::CExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto ty = cast<IntegerType>(op.getType());
    Value res =
        ty.isSigned() ? expSigned(op, adaptor, r) : expUnsigned(op, adaptor, r);
    r.replaceOp(op, res);

    return success();
  }
};

} // namespace

template <bool isLeftShift>
static Value genYulShiftOp(ConversionPatternRewriter &r, Location loc,
                           Value val, Value shiftVal, bool isSigned) {
  mlir::solgen::BuilderExt bExt(r, loc);

  unsigned resWidth = cast<IntegerType>(val.getType()).getWidth();

  // Yul shift ops work with i256.
  Value val256 = bExt.genIntCast(256, isSigned, val);
  Value shift256 = bExt.genIntCast(256, false, shiftVal);

  Value shifted256;
  if constexpr (isLeftShift) {
    shifted256 = r.create<yul::ShlOp>(loc, shift256, val256);
  } else {
    if (isSigned)
      shifted256 = r.create<yul::SarOp>(loc, shift256, val256);
    else
      shifted256 = r.create<yul::ShrOp>(loc, shift256, val256);
  }

  // Cast back to the original bitwidth.
  return bExt.genIntCast(resWidth, isSigned, shifted256);
}

template <typename OpT>
static Value getCryptoHashLowering(OpT op, uint32_t preCompieAddr,
                                   typename OpT::Adaptor adaptor,
                                   ConversionPatternRewriter &r) {
  Location loc = op.getLoc();
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(r, loc);

  Type ty = op.getData().getType();
  sol::DataLocation dataLoc = sol::getDataLocation(ty);
  assert(dataLoc == sol::DataLocation::Memory);

  Value zero = bExt.genI256Const(0);
  Value retSize = bExt.genI256Const(32);
  Value dataLen = evmB.genLoad(adaptor.getData(), dataLoc);
  Value dataSlot = evmB.genDataAddrPtr(adaptor.getData(), dataLoc);

  Value gas = r.create<mlir::yul::GasOp>(loc);
  // TODO: Solc (in YUL mode) copyies the input array one more time,
  // so we end up having two copys in heap. Also it doesn the following
  // store: mstore(add(dstAddr, dataLen), 0).
  // It's not clear why do we need this.
  // Hashing functions store their result in scratch space (0x00–0x3f).
  mlir::Value status =
      r.create<yul::StaticCallOp>(loc, gas,
                                  /*address=*/bExt.genI256Const(preCompieAddr),
                                  /*inpOffset=*/dataSlot, dataLen,
                                  /*outOffset=*/zero, /*outSize=*/retSize);

  auto statusIsZero = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              status, bExt.genI256Const(0));
  evmB.genForwardingRevert(statusIsZero);

  auto res = evmB.genLoad(zero, dataLoc);
  // For ripemd160 the result must be shifted left by 96 bits.
  if constexpr (std::is_same_v<std::decay_t<OpT>, sol::Ripemd160Op>)
    res = r.create<yul::ShlOp>(loc, bExt.genI256Const(96), res);

  return res;
}

template <typename ModOpT>
static Value genYulModOp(ConversionPatternRewriter &r, Location loc, Value x,
                         Value y, Value mod) {
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(r, loc);

  // Yul mod ops work with unsigned i256 values.
  Value x256 = bExt.genIntCast(256, /*isSigned=*/false, x);
  Value y256 = bExt.genIntCast(256, /*isSigned=*/false, y);
  Value mod256 = bExt.genIntCast(256, /*isSigned=*/false, mod);

  auto zero = bExt.genI256Const(0, loc);
  auto modEqZero =
      r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mod, zero);
  evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, modEqZero);

  return r.create<ModOpT>(loc, x256, y256, mod256);
}

namespace {

struct ShlOpLowering : public OpConversionPattern<sol::ShlOp> {
  using OpConversionPattern<sol::ShlOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    auto ty = cast<IntegerType>(op.getType());

    Value result = genYulShiftOp</*isLeftShift=*/true>(
        r, loc, adaptor.getLhs(), adaptor.getRhs(), ty.isSigned());

    r.replaceOp(op, result);
    return success();
  }
};

struct ShrOpLowering : public OpConversionPattern<sol::ShrOp> {
  using OpConversionPattern<sol::ShrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ShrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    auto ty = cast<IntegerType>(op.getType());

    Value result = genYulShiftOp</*isLeftShift=*/false>(
        r, loc, adaptor.getLhs(), adaptor.getRhs(), ty.isSigned());

    r.replaceOp(op, result);
    return success();
  }
};

struct CAddOpLowering : public OpConversionPattern<sol::CAddOp> {
  using OpConversionPattern<sol::CAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Unlike via-ir, small int (< i256) arithmetic is "legalized" by the llvm
    // backend, so we don't need a different codegen for its overflow/underflow
    // check since the legalized arithmetic works as if the small int is native
    // to evm.

    Value sum = r.create<arith::AddIOp>(loc, lhs, rhs);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // overflow, if y >= 0 and sum < x
      // underflow, if y < 0 and sum >= x
      //
      // We compare rhs with zero since the canonicalizer could make the rhs a
      // constant which would enable the arith dialect to optimize away the
      // comparison.

      auto zero = bExt.genConst(0, ty.getWidth());

      // Generate the overflow condition.
      auto rhsGtEqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rhs, zero);
      auto sumLtLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, sum, lhs);
      auto overflowCond = r.create<arith::AndIOp>(loc, rhsGtEqZero, sumLtLhs);

      // Generate the underflow condition.
      auto rhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
      auto sumGtEqLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, sum, lhs);
      auto underflowCond = r.create<arith::AndIOp>(loc, rhsLtZero, sumGtEqLhs);

      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::OrIOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(
          mlir::evm::PanicCode::UnderOverflow,
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, lhs, sum));
    }

    r.replaceOp(op, sum);
    return success();
  }
};

struct CSubOpLowering : public OpConversionPattern<sol::CSubOp> {
  using OpConversionPattern<sol::CSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value diff = r.create<arith::SubIOp>(loc, lhs, rhs);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if y >= 0 and diff > x
      // overflow, if y < 0 and diff < x

      auto zero = bExt.genConst(0, ty.getWidth());

      // Generate the overflow condition.
      auto rhsGtEqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, rhs, zero);
      auto diffGtLhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, diff, lhs);
      auto overflowCond = r.create<arith::AndIOp>(loc, rhsGtEqZero, diffGtLhs);

      // Generate the underflow condition.
      auto rhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, rhs, zero);
      auto diffLtRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff, lhs);
      auto underflowCond = r.create<arith::AndIOp>(loc, rhsLtZero, diffLtRhs);

      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::OrIOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(
          mlir::evm::PanicCode::UnderOverflow,
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, diff, lhs));
    }

    r.replaceOp(op, diff);
    return success();
  }
};

struct CMulOpLowering : public OpConversionPattern<sol::CMulOp> {
  using OpConversionPattern<sol::CMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    Value product = r.create<arith::MulIOp>(loc, lhs, rhs);
    auto zero = bExt.genConst(0, ty.getWidth());
    auto one = bExt.genConst(1, ty.getWidth());
    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x < 0 and y == int.min
      auto lhsLtZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, zero);
      auto minVal = bExt.genConst(APInt::getSignedMinValue(ty.getWidth()));
      auto rhsEqMin =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, minVal);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsLtZero, rhsEqMin));

      // over/underflow, if x != 0 and product/x != y
      // Use a safe divisor of 1 when lhs == 0 to avoid arith.divsi UB;
      // the quotient is discarded via the lhsNeqZero guard in that case.
      // FIXME: It may be reasonable to replace DivSIOp with yul::div.
      // In this case we could simplify the check, as we can get rid of the
      // select in this case.
      auto lhsNeqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, zero);
      auto safeLhs = r.create<arith::SelectOp>(loc, lhsNeqZero, lhs, one);
      auto quotient = r.create<arith::DivSIOp>(loc, product, safeLhs);
      auto quotientNeqRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotient, rhs);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsNeqZero, quotientNeqRhs));

      // Unsigned case
    } else {
      // over/underflow, if x != 0 and product/x != y
      // Use a safe divisor of 1 when lhs == 0 to avoid arith.divui UB;
      // the quotient is discarded via the lhsNeqZero guard in that case.
      auto lhsNeqZero =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, zero);
      auto safeLhs = r.create<arith::SelectOp>(loc, lhsNeqZero, lhs, one);
      auto quotient = r.create<arith::DivUIOp>(loc, product, safeLhs);
      auto quotientNeqRhs =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, quotient, rhs);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsNeqZero, quotientNeqRhs));
    }

    r.replaceOp(op, product);
    return success();
  }
};

struct AddModOpLowering : public OpConversionPattern<sol::AddModOp> {
  using OpConversionPattern<sol::AddModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::AddModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();

    Value result = genYulModOp<yul::AddModOp>(r, loc, adaptor.getX(),
                                              adaptor.getY(), adaptor.getMod());
    r.replaceOp(op, result);

    return success();
  }
};

struct MulModOpLowering : public OpConversionPattern<sol::MulModOp> {
  using OpConversionPattern<sol::MulModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::MulModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    Value result = genYulModOp<yul::MulModOp>(r, loc, adaptor.getX(),
                                              adaptor.getY(), adaptor.getMod());
    r.replaceOp(op, result);

    return success();
  }
};

struct CDivOpLowering : public OpConversionPattern<sol::CDivOp> {
  using OpConversionPattern<sol::CDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // See comments in sol.cadd lowering on why we don't have a different
    // codegen for small ints.

    auto zero = bExt.genConst(0, ty.getWidth());
    auto rhsEqZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, zero);
    evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x == int.min and y == -1
      auto minVal = bExt.genConst(APInt::getSignedMinValue(ty.getWidth()));
      auto lhsEqMin =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, minVal);
      auto minusOne = bExt.genConst(-1, ty.getWidth());
      auto rhsEqMinusOne =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rhs, minusOne);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<arith::AndIOp>(loc, lhsEqMin, rhsEqMinusOne));
      r.replaceOpWithNewOp<arith::DivSIOp>(op, lhs, rhs);

    } else {
      r.replaceOpWithNewOp<arith::DivUIOp>(op, lhs, rhs);
    }

    return success();
  }
};

struct Keccak256OpLowering : public OpConversionPattern<sol::Keccak256Op> {
  using OpConversionPattern<sol::Keccak256Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::Keccak256Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type ty = op.getAddr().getType();
    sol::DataLocation dataLoc = sol::getDataLocation(ty);

    Value dataLen = evmB.genLoad(adaptor.getAddr(), dataLoc);
    auto dataSlot = evmB.genDataAddrPtr(adaptor.getAddr(), ty);
    r.replaceOpWithNewOp<yul::Keccak256Op>(op, dataSlot, dataLen);

    return success();
  }
};

struct Sha256OpLowering : public OpConversionPattern<sol::Sha256Op> {
  using OpConversionPattern<sol::Sha256Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::Sha256Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOp(op, getCryptoHashLowering(op, /*preCompieAddr=*/2, adaptor, r));

    return success();
  }
};

struct Ripemd160OpLowering : public OpConversionPattern<sol::Ripemd160Op> {
  using OpConversionPattern<sol::Ripemd160Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::Ripemd160Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOp(op, getCryptoHashLowering(op, /*preCompieAddr=*/3, adaptor, r));

    return success();
  }
};

struct EcrecoverOpLowering : public OpConversionPattern<sol::EcrecoverOp> {
  using OpConversionPattern<sol::EcrecoverOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EcrecoverOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value zero = bExt.genI256Const(0);
    Value retSize = bExt.genI256Const(32);
    Value freePtr = evmB.genFreePtr();
    // Since the size of the encoded arguments is known in advance,
    // we ignore the return value of genABITupleEncoding.
    Value paramsSize = bExt.genI256Const(128);
    evmB.genABITupleEncoding(op.getOperandTypes(), adaptor.getOperands(),
                             freePtr);
    evmB.genFreePtrUpd(freePtr, paramsSize);

    // Hashing functions store their result in scratch space (0x00–0x3f).
    Value gas = r.create<mlir::yul::GasOp>(loc);
    // TODO: It's not clear why do we need this.
    r.create<yul::MStoreOp>(loc, zero, zero);
    mlir::Value status =
        r.create<yul::StaticCallOp>(loc, gas, /*address=*/bExt.genI256Const(1),
                                    /*inpOffset=*/freePtr, paramsSize,
                                    /*outOffset=*/zero, /*outSize=*/retSize);

    auto statusIsZero = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                status, bExt.genI256Const(0));
    evmB.genForwardingRevert(statusIsZero);
    auto res = evmB.genLoad(zero, sol::DataLocation::Memory);
    r.replaceOp(op, res);

    return success();
  }
};

struct CmpOpLowering : public OpConversionPattern<sol::CmpOp> {
  using OpConversionPattern<sol::CmpOp>::OpConversionPattern;

  arith::CmpIPredicate getSignlessPred(sol::CmpPredicate pred,
                                       bool isSigned) const {
    // Sign insensitive predicates.
    switch (pred) {
    case sol::CmpPredicate::eq:
      return arith::CmpIPredicate::eq;
    case sol::CmpPredicate::ne:
      return arith::CmpIPredicate::ne;
    default:
      break;
    }

    // Sign sensitive predicates.
    if (isSigned) {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return arith::CmpIPredicate::slt;
      case sol::CmpPredicate::le:
        return arith::CmpIPredicate::sle;
      case sol::CmpPredicate::gt:
        return arith::CmpIPredicate::sgt;
      case sol::CmpPredicate::ge:
        return arith::CmpIPredicate::sge;
      default:
        break;
      }
    } else {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return arith::CmpIPredicate::ult;
      case sol::CmpPredicate::le:
        return arith::CmpIPredicate::ule;
      case sol::CmpPredicate::gt:
        return arith::CmpIPredicate::ugt;
      case sol::CmpPredicate::ge:
        return arith::CmpIPredicate::uge;
      default:
        break;
      }
    }
    llvm_unreachable("Invalid predicate");
  }

  LogicalResult matchAndRewrite(sol::CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    Type cmpTy = op.getLhs().getType();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    bool isSigned = false;

    if (auto intTy = dyn_cast<IntegerType>(cmpTy)) {
      isSigned = intTy.isSigned();
    } else if (auto bytesTy = dyn_cast<sol::BytesType>(cmpTy)) {
      unsigned cmpWidth = bytesTy.getSize() * 8;
      if (cmpWidth < 256) {
        Value shiftAmt = bExt.genI256Const(256 - cmpWidth);
        lhs = r.create<arith::ShRUIOp>(loc, lhs, shiftAmt);
        rhs = r.create<arith::ShRUIOp>(loc, rhs, shiftAmt);
      }
    } else if (sol::isAddressLikeType(cmpTy)) {
      APInt mask = APInt::getLowBitsSet(256, 160);
      lhs = r.create<arith::AndIOp>(loc, lhs, bExt.genI256Const(mask));
      rhs = r.create<arith::AndIOp>(loc, rhs, bExt.genI256Const(mask));
    } else if (!isa<sol::EnumType>(cmpTy)) {
      llvm_unreachable("Unexpected type for comparison");
    }

    arith::CmpIPredicate signlessPred =
        getSignlessPred(op.getPredicate(), isSigned);
    r.replaceOpWithNewOp<arith::CmpIOp>(op, signlessPred, lhs, rhs);
    return success();
  }
};

struct AllocaOpLowering : public OpConversionPattern<sol::AllocaOp> {
  using OpConversionPattern<sol::AllocaOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());

    auto ptrTy = cast<sol::PointerType>(op.getAllocType());
    Type convertedEltTy =
        getTypeConverter()->convertType(ptrTy.getPointeeType());

    r.replaceOpWithNewOp<LLVM::AllocaOp>(
        op, LLVM::LLVMPointerType::get(r.getContext()), convertedEltTy,
        bExt.genI256Const(1));
    return success();
  }
};

struct MallocOpLowering : public OpConversionPattern<sol::MallocOp> {
  using OpConversionPattern<sol::MallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::MallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());
    r.replaceOp(op, evmB.genMemAlloc(op.getType(), op.getZeroInit(), {},
                                     adaptor.getSize()));
    return success();
  }
};

struct ArrayLitOpLowering : public OpConversionPattern<sol::ArrayLitOp> {
  using OpConversionPattern<sol::ArrayLitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ArrayLitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    r.replaceOp(op,
                evmB.genMemAlloc(op.getType(), false, adaptor.getIns(), {}));
    return success();
  }
};

struct StringLitOpLowering : public OpConversionPattern<sol::StringLitOp> {
  using OpConversionPattern<sol::StringLitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StringLitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {

    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    mlir::solgen::BuilderExt bExt(r, op.getLoc());

    StringRef lit = adaptor.getValue();
    Value litSize = bExt.genI256Const(lit.size());

    // Beyond a certain length, string literals are cheaper to embed as in-code
    // data and load via CODECOPY than to construct directly.
    // TODO: determine the exact size threshold.
    Value allocPtr = nullptr;
    if (lit.size() > 128) {
      // genMemAlloc(Type) calls genMemAllocForDynArray which writes the length
      // word. Then the CODECOPY below fills only the data area.
      allocPtr = evmB.genMemAlloc(op.getType(), false, {}, litSize, loc);
      auto mod = op->getParentOfType<ModuleOp>();
      // Create a global constant initialized with the given string literal.
      LLVM::GlobalOp globConst = bExt.getStringLiteralGlobalOp(lit, mod);
      auto ptrToArray =
          LLVM::LLVMPointerType::get(mod.getContext(), /*addressSpace=*/4);
      auto addrOf =
          r.create<LLVM::AddressOfOp>(loc, ptrToArray, globConst.getSymName());
      Value intAddr =
          r.create<LLVM::PtrToIntOp>(loc, r.getIntegerType(256), addrOf);
      Value strPtr =
          evmB.genDataAddrPtr(allocPtr, sol::DataLocation::Memory, loc);
      r.create<yul::CodeCopyOp>(loc, strPtr, intAddr, litSize);
    } else {
      // Allocate raw memory (32-byte length slot + rounded-up data) and rely on
      // genStringStore to write both the length word and the data. This avoids
      // the redundant length write that genMemAlloc(Type) would produce via
      // genMemAllocForDynArray.
      size_t roundedLen = llvm::alignTo(lit.size(), 32u);
      allocPtr = evmB.genMemAlloc(bExt.genI256Const(32 + roundedLen), loc);
      evmB.genStringStore(lit.str(), allocPtr, loc);
    }

    r.replaceOp(op, allocPtr);
    return success();
  }
};

struct ConcatOpLowering : public OpConversionPattern<sol::ConcatOp> {
  using OpConversionPattern<sol::ConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Value freePtr = evmB.genFreePtr();
    Value dstStart =
        r.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(32));

    Value currDst = dstStart;
    for (auto [origSrc, src] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
      Type ty = origSrc.getType();
      if (auto bytesTy = dyn_cast<sol::BytesType>(ty)) {
        // bytesN values are left-aligned in an i256 with trailing bytes zeroed
        // by invariant. On the other hand, solc applies maks on the trailing
        // bytes. See solx-solidity, #78.
        r.create<yul::MStoreOp>(loc, currDst, src);
        Value length = bExt.genI256Const(bytesTy.getSize());
        currDst = r.create<arith::AddIOp>(loc, currDst, length);
        continue;
      }

      assert(isa<sol::StringType>(ty));
      Value length = nullptr;
      sol::DataLocation srcDataLoc = sol::getDataLocation(ty);
      Value srcDataAddr = evmB.genDataAddrPtr(src, srcDataLoc, loc);
      if (srcDataLoc == sol::DataLocation::Memory) {
        // srcAddr points to the length word, srcDataAddr = srcAddr + 32.
        // Use srcAddr so genDynSize reads mload(srcAddr) = the length word.
        length = evmB.genDynSize(src, ty, loc);
        r.create<yul::MCopyOp>(loc, currDst, srcDataAddr, length);
      } else if (srcDataLoc == sol::DataLocation::CallData) {
        // srcDataAddr is an i256 data pointer. Use srcAddr (fat pointer).
        length = evmB.genDynSize(src, ty, loc);
        r.create<yul::CallDataCopyOp>(loc, currDst, srcDataAddr, length);
      } else {
        // srcAddr is the storage slot, srcDataAddr = keccak256(srcAddr).
        // Decode the encoded length from the slot value and copy the raw
        // data bytes directly into the concat buffer (no length word here).
        Value lengthSlot = evmB.genLoad(src, srcDataLoc, loc);
        length = evmB.genStringLength(lengthSlot, srcDataLoc, loc);
        evmB.genCopyStringDataToMemory(srcDataAddr, lengthSlot, length, currDst,
                                       loc);
      }
      currDst = r.create<arith::AddIOp>(loc, currDst, length);
    }
    Value dataSize = r.create<arith::SubIOp>(loc, currDst, dstStart);
    r.create<yul::MStoreOp>(loc, freePtr, dataSize);
    Value allocationSize = r.create<arith::SubIOp>(loc, currDst, freePtr);
    evmB.genFreePtrUpd(freePtr, bExt.genRoundUpToMultiple<32>(allocationSize));
    r.replaceOp(op, freePtr);

    return success();
  }
};

struct GetCallDataOpLowering : public OpConversionPattern<sol::GetCallDataOp> {
  using OpConversionPattern<sol::GetCallDataOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GetCallDataOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());
    Value zero = bExt.genI256Const(0);
    if (auto strTy = dyn_cast<sol::StringType>(op.getType())) {
      if (strTy.getDataLocation() == sol::DataLocation::CallData) {
        Value callDataSz = r.create<yul::CallDataSizeOp>(op.getLoc());
        r.replaceOp(op, bExt.genLLVMStruct({zero, callDataSz}));
        return success();
      }
    }
    r.replaceOp(op, zero);
    return success();
  }
};

struct PushOpLowering : public OpConversionPattern<sol::PushOp> {
  using OpConversionPattern<sol::PushOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Type inpTy = op.getInp().getType();
    if (isa<sol::StringType>(inpTy)) {
      r.replaceOp(op, evmB.genPushVoidToString(adaptor.getInp(), loc));
      return success();
    }

    Value slot = adaptor.getInp();
    Value oldSize = r.create<yul::SLoadOp>(loc, slot);
    Value newSize = r.create<arith::AddIOp>(loc, oldSize, bExt.genI256Const(1));
    r.create<yul::SStoreOp>(loc, slot, newSize);
    Value dataSlot = evmB.genDataAddrPtr(slot, sol::DataLocation::Storage);

    // Get element type from the input type.
    Type eltTy;
    if (auto arrTy = dyn_cast<sol::ArrayType>(inpTy)) {
      eltTy = arrTy.getEltType();
    } else {
      llvm_unreachable("");
    }

    if (evm::canBePacked(eltTy)) {
      r.replaceOp(op, evmB.genPackedStorageAddr(dataSlot, oldSize, eltTy));
    } else {
      // Slot-aligned layout.
      Value stride = bExt.genI256Const(evm::getStorageSlotCount(eltTy));
      Value scaledIdx = r.create<arith::MulIOp>(loc, oldSize, stride);
      r.replaceOp(op, r.create<arith::AddIOp>(loc, dataSlot, scaledIdx));
    }
    return success();
  }
};

struct PushStringOpLowering : public OpConversionPattern<sol::PushStringOp> {
  using OpConversionPattern<sol::PushStringOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PushStringOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type inpValueTy = op.getValue().getType();
    Value castedVal =
        bExt.genIntCast(256, /*isSigned*/ false, adaptor.getValue());

    // Solc clears the most significant bytes using and 0xff, however LLVM DAG
    // legalization handles this automatically when legalizing zext i8 to i256.
    Value byte =
        isa<sol::BytesType>(inpValueTy)
            ? r.create<arith::ShRUIOp>(loc, castedVal, bExt.genI256Const(248))
                  .getResult()
            : castedVal;
    evmB.genPushToString(adaptor.getAddr(), byte, loc);
    r.eraseOp(op);

    return success();
  }
};

struct PopOpLowering : public OpConversionPattern<sol::PopOp> {
  using OpConversionPattern<sol::PopOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PopOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Type inpTy = op.getInp().getType();
    Value slot = adaptor.getInp();
    Value data = evmB.genLoad(slot, sol::DataLocation::Storage, loc);
    Value oldSize =
        isa<sol::StringType>(inpTy)
            ? evmB.genStringLength(data, sol::DataLocation::Storage, loc)
            : data;

    // Generate the empty array panic check.
    Value panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              oldSize, bExt.genI256Const(0));
    evmB.genPanic(mlir::evm::PanicCode::EmptyArrayPop, panicCond);

    if (isa<sol::StringType>(inpTy)) {
      evmB.genPopString(slot, data, oldSize, loc);
      r.eraseOp(op);
      return success();
    }

    Value newSize = r.create<arith::SubIOp>(loc, oldSize, bExt.genI256Const(1));
    r.create<yul::SStoreOp>(loc, slot, newSize);
    Value dataSlot = evmB.genDataAddrPtr(slot, sol::DataLocation::Storage);

    // Get element type from the input type.
    Type eltTy;
    if (auto arrTy = dyn_cast<sol::ArrayType>(inpTy)) {
      eltTy = arrTy.getEltType();
    } else {
      llvm_unreachable("");
    }

    // Zero the deleted element.
    if (evm::canBePacked(eltTy)) {
      // Packed: load-modify-store to clear just those bytes.
      unsigned numBits = evm::getStorageByteSize(eltTy) * 8;
      Value eltByteSize = bExt.genI256Const(evm::getStorageByteSize(eltTy));
      Value bytePos = r.create<arith::MulIOp>(loc, newSize, eltByteSize);
      Value thirtyTwo = bExt.genI256Const(32);
      Value slotOffset = r.create<arith::DivUIOp>(loc, bytePos, thirtyTwo);
      Value byteOffset = r.create<arith::RemUIOp>(loc, bytePos, thirtyTwo);
      Value tailSlot = r.create<arith::AddIOp>(loc, dataSlot, slotOffset);

      Value shiftBits =
          r.create<arith::MulIOp>(loc, byteOffset, bExt.genI256Const(8));
      Value cleared = evmB.genPunchHole(tailSlot, shiftBits, numBits);
      r.create<yul::SStoreOp>(loc, tailSlot, cleared);
    } else {
      // Slot-aligned: zero the whole slot.
      Value stride = bExt.genI256Const(evm::getStorageSlotCount(eltTy));
      Value scaledIdx = r.create<arith::MulIOp>(loc, newSize, stride);
      Value tailAddr = r.create<arith::AddIOp>(loc, dataSlot, scaledIdx);
      r.create<yul::SStoreOp>(loc, tailAddr, bExt.genI256Const(0));
    }

    r.eraseOp(op);
    return success();
  }
};

struct AddrOfOpLowering : public OpRewritePattern<sol::AddrOfOp> {
  using OpRewritePattern<sol::AddrOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::AddrOfOp op,
                                PatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());

    auto parentContract = op->getParentOfType<sol::ContractOp>();
    auto *sym = parentContract.lookupSymbol(op.getVar());
    assert(sym);
    if (auto stateVarOp = dyn_cast<sol::StateVarOp>(sym)) {
      Value slot = bExt.genI256Const(stateVarOp.getSlot());
      if (evm::canBePacked(stateVarOp.getType())) {
        Value offset = bExt.genI256Const(stateVarOp.getByteOffset());
        r.replaceOp(op, bExt.genLLVMStruct({slot, offset}));
      } else {
        r.replaceOp(op, slot);
      }
      return success();
    }
    auto immOp = cast<sol::ImmutableOp>(sym);
    assert(immOp->hasAttr("addr"));
    IntegerAttr addr = cast<IntegerAttr>(immOp->getAttr("addr"));
    r.replaceOp(op, bExt.genI256Const(addr.getValue()));
    return success();
  }
};

struct GepOpLowering : public OpConversionPattern<sol::GepOp> {
  using OpConversionPattern<sol::GepOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GepOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type baseAddrTy = op.getBaseAddr().getType();
    Value remappedBaseAddr = adaptor.getBaseAddr();
    Value idx = adaptor.getIdx();
    sol::DataLocation dataLoc = sol::getDataLocation(baseAddrTy);
    Value res;

    switch (dataLoc) {
    case sol::DataLocation::Stack: {
      auto stkPtrTy =
          LLVM::LLVMPointerType::get(r.getContext(), evm::AddrSpace_Stack);
      res = r.create<LLVM::GEPOp>(loc, /*resultType=*/stkPtrTy,
                                  /*basePtrType=*/remappedBaseAddr.getType(),
                                  remappedBaseAddr, idx);
      break;
    }

    case sol::DataLocation::Storage: {
      // Storage array
      if (auto arrTy = dyn_cast<sol::ArrayType>(baseAddrTy)) {
        Type eltTy = arrTy.getEltType();
        assert((isa<IntegerType>(eltTy) || sol::isNonPtrRefType(eltTy)) &&
               "NYI");

        // Bounds check (always for dynamic, non-const index for static).
        auto idxTy = cast<IntegerType>(idx.getType());
        Value castedIdx = bExt.genIntCast(256, idxTy.isSigned(), idx);
        bool isConstIdx = !isa<BlockArgument>(idx) &&
                          isa<arith::ConstantIntOp>(idx.getDefiningOp());
        if (arrTy.isDynSized() || !isConstIdx) {
          Value size = arrTy.isDynSized()
                           ? evmB.genDynSize(remappedBaseAddr, baseAddrTy)
                           : bExt.genI256Const(arrTy.getSize());
          auto panicCond = r.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, castedIdx, size);
          evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);
        }

        // Base slot: keccak256(slot) for dynamic, slot for static.
        Value baseSlot = arrTy.isDynSized()
                             ? evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy)
                             : remappedBaseAddr;

        if (evm::canBePacked(eltTy)) {
          res = evmB.genPackedStorageAddr(baseSlot, castedIdx, eltTy);
        } else {
          // Slot-aligned layout.
          Value stride = bExt.genI256Const(evm::getStorageSlotCount(eltTy));
          Value scaledIdx = r.create<arith::MulIOp>(loc, castedIdx, stride);
          res = r.create<arith::AddIOp>(loc, baseSlot, scaledIdx);
        }

      } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
        // Field index is always constant for struct access.
        auto constIdx = cast<arith::ConstantIntOp>(idx.getDefiningOp());
        int64_t fieldIdx = constIdx.value();

        ArrayRef<Type> memberTypes = structTy.getMemberTypes();
        Type fieldTy = memberTypes[fieldIdx];

        // Compute slot/byte offset by walking preceding fields.
        unsigned slotOffset = 0;
        unsigned byteOffset = 0;

        for (int64_t i = 0; i < fieldIdx; ++i) {
          Type prevTy = memberTypes[i];

          if (evm::canBePacked(prevTy)) {
            unsigned prevSize = evm::getStorageByteSize(prevTy);
            if (byteOffset + prevSize > 32) {
              slotOffset++;
              byteOffset = 0;
            }
            byteOffset += prevSize;
          } else {
            // Non-packable: flush partial slot, add its slot count.
            if (byteOffset > 0) {
              slotOffset++;
              byteOffset = 0;
            }
            slotOffset += evm::getStorageSlotCount(prevTy);
          }
        }

        // Position for target field.
        if (evm::canBePacked(fieldTy)) {
          unsigned fieldSize = evm::getStorageByteSize(fieldTy);
          if (byteOffset + fieldSize > 32) {
            slotOffset++;
            byteOffset = 0;
          }
          // Result: {baseSlot + slotOffset, byteOffset}
          Value slot = r.create<arith::AddIOp>(loc, remappedBaseAddr,
                                               bExt.genI256Const(slotOffset));
          res = bExt.genLLVMStruct({slot, bExt.genI256Const(byteOffset)});
        } else {
          // Non-packable: flush partial slot.
          if (byteOffset > 0)
            slotOffset++;
          res = r.create<arith::AddIOp>(loc, remappedBaseAddr,
                                        bExt.genI256Const(slotOffset));
        }

      } else if (isa<sol::StringType>(baseAddrTy)) {
        res = evmB.genStringItemAddress(remappedBaseAddr, idx, loc);
      } else {
        llvm_unreachable("NYI");
      }

      assert(res);
      break;
    }

    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory: {
      // Memory/calldata array
      if (auto arrTy = dyn_cast<sol::ArrayType>(baseAddrTy)) {
        Value addrAtIdx;

        Type eltTy = arrTy.getEltType();
        (void)eltTy;
        assert((isa<IntegerType>(eltTy) || sol::isNonPtrRefType(eltTy)) &&
               "NYI");

        // Don't generate out-of-bounds check for constant indexing of static
        // arrays.
        if (!isa<BlockArgument>(idx) &&
            isa<arith::ConstantIntOp>(idx.getDefiningOp())) {
          auto constIdx = cast<arith::ConstantIntOp>(idx.getDefiningOp());
          if (!arrTy.isDynSized()) {
            // FIXME: Should this be done by the verifier?
            assert(constIdx.value() < arrTy.getSize());
            addrAtIdx = r.create<arith::AddIOp>(
                loc, remappedBaseAddr,
                bExt.genI256Const(constIdx.value() * 32));
          }
        }

        if (!addrAtIdx) {
          //
          // Generate PanicCode::ArrayOutOfBounds check.
          //
          Value size;
          if (arrTy.isDynSized())
            size = evmB.genDynSize(remappedBaseAddr, baseAddrTy);
          else
            size = bExt.genI256Const(arrTy.getSize());

          // Generate `if iszero(lt(index, <arrayLen>(baseRef)))` (yul).
          auto idxTy = cast<IntegerType>(idx.getType());
          Value castedIdx = bExt.genIntCast(256, idxTy.isSigned(), idx);
          auto panicCond = r.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::uge, castedIdx, size);
          evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

          //
          // Generate the address.
          //
          Value stride = bExt.genI256Const(32);
          Value scaledIdx = r.create<arith::MulIOp>(loc, castedIdx, stride);
          if (arrTy.isDynSized()) {
            Value dataAddr = evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy);
            addrAtIdx = r.create<arith::AddIOp>(loc, dataAddr, scaledIdx);
          } else {
            addrAtIdx =
                r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);
          }
        }
        assert(addrAtIdx);

        res = addrAtIdx;

        // Memory/calldata struct
      } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
#ifndef NDEBUG
        for (Type ty : structTy.getMemberTypes())
          assert(isa<IntegerType>(ty) || sol::isNonPtrRefType(ty) && "NYI");
#endif

        auto idxConstOp = cast<arith::ConstantIntOp>(idx.getDefiningOp());
        Value memberIdx =
            bExt.genIntCast(/*width=*/256, /*isSigned=*/false, idxConstOp);
        auto scaledIdx =
            r.create<arith::MulIOp>(loc, memberIdx, bExt.genI256Const(32));
        res = r.create<arith::AddIOp>(loc, remappedBaseAddr, scaledIdx);

        // Bytes (!sol.string)
      } else if (auto strTy = dyn_cast<sol::StringType>(baseAddrTy)) {
        Value size = evmB.genDynSize(remappedBaseAddr, baseAddrTy);
        Value castedIdx =
            bExt.genIntCast(/*width=*/256, /*isSigned=*/false, idx);
        auto panicCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                 castedIdx, size);
        evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

        Value dataAddr = evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy);
        res = r.create<arith::AddIOp>(loc, dataAddr, castedIdx);
      }

      assert(res);
      break;
    }

    default:
      llvm_unreachable("NYI");
      break;
    }

    r.replaceOp(op, res);
    return success();
  }
};

struct MapOpLowering : public OpConversionPattern<sol::MapOp> {
  using OpConversionPattern<sol::MapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::MapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    // Assert that the mapping is a slot (result of sol.addr_of or sol.map).
    assert(cast<IntegerType>(adaptor.getMapping().getType()).getWidth() == 256);

    // Setup arguments to keccak256.
    auto zero = bExt.genI256Const(0);
    bool keySigned = false;
    if (auto keyTy = dyn_cast<IntegerType>(op.getKey().getType()))
      keySigned = keyTy.isSigned();
    else
      assert(sol::isAddressLikeType(op.getKey().getType()) &&
             "NYI: Unsupported mapping key type");
    auto key = bExt.genIntCast(/*width=*/256, keySigned, adaptor.getKey());
    r.create<yul::MStoreOp>(loc, zero, key);
    r.create<yul::MStoreOp>(loc, bExt.genI256Const(0x20), adaptor.getMapping());

    Value slot = r.create<yul::Keccak256Op>(loc, zero, bExt.genI256Const(0x40));

    // Result is {slot, 0} or just slot depending on pointee type.
    Type resTy = op.getResult().getType();
    if (auto ptrTy = dyn_cast<sol::PointerType>(resTy);
        ptrTy && ptrTy.getDataLocation() == sol::DataLocation::Storage &&
        evm::canBePacked(ptrTy.getPointeeType())) {
      r.replaceOp(op, bExt.genLLVMStruct({slot, zero}));
    } else {
      r.replaceOp(op, slot);
    }
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<sol::LoadOp> {
  using OpConversionPattern<sol::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value addr = adaptor.getAddr();
    sol::DataLocation dataLoc = sol::getDataLocation(op.getAddr().getType());
    auto i256Ty = r.getIntegerType(256);

    switch (dataLoc) {
    case sol::DataLocation::Stack: {
      Type eltTy =
          cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();
      Type loadTy = getTypeConverter()->convertType(eltTy);
      assert(loadTy);
      r.replaceOpWithNewOp<LLVM::LoadOp>(op, loadTy, addr,
                                         evm::getAlignment(addr));
      return success();
    }
    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory: {
      auto addrTy = cast<sol::PointerType>(op.getAddr().getType());
      auto bytesEleTy = dyn_cast<sol::BytesType>(addrTy.getPointeeType());
      // If loading from `bytes`, generate the low bits mask-off of the loaded
      // value.
      if (bytesEleTy) {
        unsigned numBits = bytesEleTy.getSize() * 8;
        APInt mask(/*numBits=*/256, 0);
        assert(numBits <= 256);
        mask.setHighBits(numBits);
        auto load = evmB.genLoad(addr, dataLoc);
        r.replaceOpWithNewOp<arith::AndIOp>(op, load, bExt.genI256Const(mask));
        return success();
      }

      auto ld = evmB.genLoad(addr, dataLoc);
      if (sol::isAddressLikeType(op.getType())) {
        APInt mask = APInt::getLowBitsSet(256, 160);
        r.replaceOpWithNewOp<arith::AndIOp>(op, ld, bExt.genI256Const(mask));
        return success();
      }
      if (auto intTy = dyn_cast<IntegerType>(op.getType())) {
        Value castedRes = bExt.genIntCastWithBoolCleanup(
            intTy.getWidth(), intTy.isSigned(), ld, loc);
        r.replaceOp(op, castedRes);
        return success();
      }
      r.replaceOp(op, ld);
      return success();
    }
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient: {
      Type eltTy =
          cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();

      // Helper to emit sload or tload based on data location.
      auto genSlotLoad = [&](Value slot) -> Value {
        if (dataLoc == sol::DataLocation::Transient)
          return r.create<yul::TLoadOp>(loc, slot);
        return r.create<yul::SLoadOp>(loc, slot);
      };

      if (evm::canBePacked(eltTy)) {
        // addr is {slot, offset} struct
        Value slot = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, addr, r.getDenseI64ArrayAttr({0}));
        Value offset = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, addr, r.getDenseI64ArrayAttr({1}));

        // Load the full slot and shift right by (offset * 8) to extract.
        Value slotVal = genSlotLoad(slot);
        Value shiftBits =
            r.create<arith::MulIOp>(loc, offset, bExt.genI256Const(8));
        Value shifted = r.create<arith::ShRUIOp>(loc, slotVal, shiftBits);

        // BytesType: shl to restore high-bits alignment.
        if (auto bytesTy = dyn_cast<sol::BytesType>(eltTy)) {
          unsigned numBits = bytesTy.getSize() * 8;
          Value res = r.create<arith::ShLIOp>(loc, shifted,
                                              bExt.genI256Const(256 - numBits));
          r.replaceOp(op, res);
        } else if (sol::isAddressLikeType(op.getType())) {
          APInt mask = APInt::getLowBitsSet(256, 160);
          r.replaceOpWithNewOp<arith::AndIOp>(op, shifted,
                                              bExt.genI256Const(mask));
        } else if (auto intTy = dyn_cast<IntegerType>(op.getType())) {
          Value castedRes = bExt.genIntCastWithBoolCleanup(
              intTy.getWidth(), intTy.isSigned(), shifted, loc,
              /*maskBoolAsStorageByte=*/true);
          r.replaceOp(op, castedRes);
        } else if (isa<sol::FuncRefType>(eltTy)) {
          // FuncRef is 64 bits, mask to lower 64 bits.
          APInt mask = APInt::getLowBitsSet(256, 64);
          Value masked =
              r.create<arith::AndIOp>(loc, shifted, bExt.genI256Const(mask));
          r.replaceOp(op, masked);
        } else if (isa<sol::EnumType>(eltTy)) {
          // Enums are 1 byte, mask to lower 8 bits.
          APInt mask = APInt::getLowBitsSet(256, 8);
          Value masked =
              r.create<arith::AndIOp>(loc, shifted, bExt.genI256Const(mask));
          r.replaceOp(op, masked);
        } else {
          llvm_unreachable("NYI");
        }
      } else {
        // addr is just slot
        Value slotVal = genSlotLoad(addr);
        r.replaceOp(op, slotVal);
      }
      return success();
    }
    default: {
      llvm_unreachable("");
    }
    };
  }
};

struct LoadImmutableOpLowering
    : public OpConversionPattern<sol::LoadImmutableOp> {
  using OpConversionPattern<sol::LoadImmutableOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadImmutableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // Track the "addr" attribute from the referenced sol.immutable op.
    SmallVector<NamedAttribute> attrs = llvm::to_vector(op->getAttrs());
    auto parentContract = op->getParentOfType<sol::ContractOp>();
    Operation *sym = parentContract.lookupSymbol(op.getName());
    auto immOp = cast<sol::ImmutableOp>(sym);
    attrs.push_back(
        r.getNamedAttr("addr", immOp->getAttrOfType<IntegerAttr>("addr")));

    // Legalize result types and add the "addr" attribute.
    SmallVector<Type> newResTys;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResTys)))
      return failure();
    r.replaceOpWithNewOp<sol::LoadImmutableOp>(op, newResTys,
                                               adaptor.getOperands(), attrs);
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<sol::StoreOp> {
  using OpConversionPattern<sol::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value remappedVal = adaptor.getVal();
    Value remappedAddr = adaptor.getAddr();
    sol::DataLocation dataLoc = sol::getDataLocation(op.getAddr().getType());

    switch (dataLoc) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::StoreOp>(op, remappedVal, remappedAddr,
                                          evm::getAlignment(remappedAddr));
      return success();
    case sol::DataLocation::Immutable:
    case sol::DataLocation::Memory: {
      Type addrTy = op.getAddr().getType();
      sol::DataLocation dataLoc = sol::getDataLocation(addrTy);

      // Generate mstore8 for storing to `bytes`.
      auto bytesEleTy = dyn_cast<sol::BytesType>(sol::getEltType(addrTy));
      if (bytesEleTy && dataLoc == sol::DataLocation::Memory) {
        assert(bytesEleTy.getSize() == 1 && "NYI");
        auto byteVal =
            r.create<yul::ByteOp>(loc, bExt.genI256Const(0), remappedVal);
        r.replaceOpWithNewOp<yul::MStore8Op>(op, remappedAddr, byteVal);
        return success();
      }

      if (sol::isAddressLikeType(op.getVal().getType())) {
        APInt mask = APInt::getLowBitsSet(256, 160);
        Value canonicalAddr =
            r.create<arith::AndIOp>(loc, remappedVal, bExt.genI256Const(mask));
        evmB.genStore(canonicalAddr, remappedAddr, dataLoc);
        r.eraseOp(op);
        return success();
      }

      if (auto intTy = dyn_cast<IntegerType>(op.getVal().getType())) {
        Value castedVal =
            bExt.genIntCast(/*width=*/256, intTy.isSigned(), remappedVal);
        evmB.genStore(castedVal, remappedAddr, dataLoc);
        r.eraseOp(op);
        return success();
      }
      evmB.genStore(remappedVal, remappedAddr, dataLoc);
      r.eraseOp(op);
      return success();
    }
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient: {
      Type eltTy =
          cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();
      auto i256Ty = r.getIntegerType(256);

      // Helper to create sstore or tstore based on data location.
      auto genSlotStore = [&](Value slot, Value val) {
        if (dataLoc == sol::DataLocation::Transient)
          r.create<yul::TStoreOp>(loc, slot, val);
        else
          r.create<yul::SStoreOp>(loc, slot, val);
      };

      if (evm::canBePacked(eltTy)) {
        // addr is {slot, offset} struct
        Value slot = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, remappedAddr, r.getDenseI64ArrayAttr({0}));
        Value offset = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, remappedAddr, r.getDenseI64ArrayAttr({1}));

        // This tracks the value we prepare to be stored.
        Value preparedVal;
        unsigned numBits;
        if (auto bytesTy = dyn_cast<sol::BytesType>(eltTy)) {
          // BytesType: shr to convert from MSB-aligned to LSB-aligned.
          numBits = bytesTy.getSize() * 8;
          preparedVal = r.create<arith::ShRUIOp>(
              loc, remappedVal, bExt.genI256Const(256 - numBits));
        } else if (sol::isAddressLikeType(op.getVal().getType())) {
          numBits = 160;
          APInt mask = APInt::getLowBitsSet(256, numBits);
          preparedVal = r.create<arith::AndIOp>(loc, remappedVal,
                                                bExt.genI256Const(mask));
        } else if (auto intTy = dyn_cast<IntegerType>(op.getVal().getType())) {
          // In storage packing, bool occupies 1 byte not a single bit.
          numBits = intTy.getWidth() == 1 ? 8 : intTy.getWidth();
          // zext to i256
          preparedVal =
              bExt.genIntCast(/*width=*/256, /*isSigned=*/false, remappedVal);
        } else if (isa<sol::FuncRefType>(eltTy)) {
          // FuncRef is 64 bits, already i256.
          numBits = 64;
          preparedVal = remappedVal;
        } else if (isa<sol::EnumType>(eltTy)) {
          // Enums can have at most 256 members, so always 1 byte.
          numBits = 8;
          preparedVal =
              bExt.genIntCast(/*width=*/256, /*isSigned=*/false, remappedVal);
        } else {
          llvm_unreachable("NYI");
        }

        // Punch hole in slot for new value.
        Value shiftBits =
            r.create<arith::MulIOp>(loc, offset, bExt.genI256Const(8));
        Value slotWithHole =
            evmB.genPunchHole(slot, shiftBits, numBits, dataLoc);

        // Shift new value to position: shl(offset * 8, preparedVal)
        preparedVal = r.create<arith::ShLIOp>(loc, preparedVal, shiftBits);

        // Combine, i.e. or(slotWithHole, preparedVal) and store.
        genSlotStore(slot,
                     r.create<arith::OrIOp>(loc, slotWithHole, preparedVal));
      } else {
        // addr is just slot, do direct store.
        genSlotStore(remappedAddr, remappedVal);
      }
      r.eraseOp(op);
      return success();
    }
    default:
      break;
    };

    llvm_unreachable("NYI: Calldata data-location");
  }
};

struct DataLocCastOpLowering : public OpConversionPattern<sol::DataLocCastOp> {
  using OpConversionPattern<sol::DataLocCastOp>::OpConversionPattern;

  Value genCopy(Value srcAddr, Type ty, PatternRewriter &r,
                Location loc) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    sol::DataLocation srcDataLoc = sol::DataLocation::Storage;
    sol::DataLocation dstDataLoc = sol::DataLocation::Memory;
    assert((srcDataLoc == sol::DataLocation::Storage &&
            dstDataLoc == sol::DataLocation::Memory) &&
           "FIXME");

    if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
      Value size, dstAddr, dstDataAddr, srcDataAddr;
      if (arrTy.isDynSized()) {
        size = evmB.genLoad(srcAddr, srcDataLoc);
        dstAddr = evmB.genMemAllocForDynArray(
            size, r.create<arith::MulIOp>(loc, size, bExt.genI256Const(32)));
        dstDataAddr = evmB.genDataAddrPtr(dstAddr, dstDataLoc);
        srcDataAddr = evmB.genDataAddrPtr(srcAddr, srcDataLoc);
      } else {
        size = bExt.genI256Const(arrTy.getSize());
        dstAddr = evmB.genMemAlloc(arrTy.getSize() * 32);
        dstDataAddr = dstAddr;
        srcDataAddr = srcAddr;
      }
      r.create<scf::ForOp>(
          loc, /*lowerBound=*/bExt.genIdxConst(0),
          /*upperBound=*/bExt.genCastToIdx(size),
          /*step=*/bExt.genIdxConst(1),
          /*initArgs=*/ValueRange{},
          /*builder=*/
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange initArgs) {
            Value i256IndVar = bExt.genCastToI256(indVar);
            Value srcAddrI = evmB.genAddrAtIdx(srcDataAddr, i256IndVar, arrTy,
                                               srcDataLoc, loc);
            Value dstAddrI = evmB.genAddrAtIdx(dstDataAddr, i256IndVar, arrTy,
                                               dstDataLoc, loc);
            evmB.genStore(genCopy(srcAddrI, arrTy.getEltType(), r, loc),
                          dstAddrI, dstDataLoc);
            r.create<scf::YieldOp>(loc);
          });
      return dstAddr;
    }

    assert(isa<IntegerType>(ty) && "NYI");
    return evmB.genLoad(srcAddr, srcDataLoc);
  }

  LogicalResult matchAndRewrite(sol::DataLocCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type srcTy = op.getInp().getType();
    Type dstTy = op.getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);

    if (dstDataLoc == sol::DataLocation::Memory) {
      // String type
      if (isa<sol::StringType>(srcTy)) {
        if (srcDataLoc == sol::DataLocation::CallData) {
          Value sizeInBytes = evmB.genDynSize(adaptor.getInp(), srcTy);
          Value memAddr = evmB.genMemAllocForDynArray(
              sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes));
          Value srcDataAddr =
              evmB.genDataAddrPtr(adaptor.getInp(), srcDataLoc, loc);
          Value dstDataAddr = evmB.genDataAddrPtr(memAddr, dstDataLoc, loc);

          r.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr,
                                        sizeInBytes);
          r.replaceOp(op, memAddr);
          return success();
        }
        if (srcDataLoc == sol::DataLocation::Storage) {
          // Generate the memory allocation.
          Value sizeSlot = evmB.genLoad(adaptor.getInp(), srcDataLoc, loc);
          Value sizeInBytes = evmB.genStringLength(sizeSlot, srcDataLoc, loc);
          Value dstAddr = evmB.genMemAllocForDynArray(
              sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes));

          Value srcDataPtr =
              evmB.genDataAddrPtr(adaptor.getInp(), srcDataLoc, loc);
          evmB.genCopyStringToMemory(srcDataPtr, sizeSlot, sizeInBytes, dstAddr,
                                     loc);
          r.replaceOp(op, dstAddr);
          return success();
        }
      }

      if (isa<sol::ArrayType>(srcTy)) {
        r.replaceOp(op, genCopy(adaptor.getInp(), srcTy, r, loc));
        return success();
      }
    }

    llvm_unreachable("NYI");
  }
};

struct LengthOpLowering : public OpConversionPattern<sol::LengthOp> {
  using OpConversionPattern<sol::LengthOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LengthOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type ty = op.getInp().getType();

    if (auto stringTy = dyn_cast<sol::StringType>(ty)) {
      r.replaceOp(op, evmB.genDynSize(adaptor.getInp(), ty));
      return success();
    }
    if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
      if (arrTy.isDynSized()) {
        r.replaceOp(op, evmB.genDynSize(adaptor.getInp(), ty));
        return success();
      }
      r.replaceOp(op, bExt.genI256Const(arrTy.getSize()));
      return success();
    }
    llvm_unreachable("NYI");
  }
};

struct SliceOpLowering : public OpConversionPattern<sol::SliceOp> {
  using OpConversionPattern<sol::SliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type arrTy = op.getArr().getType();
    assert(sol::getDataLocation(arrTy) == sol::DataLocation::CallData);

    Value arr = adaptor.getArr();
    Value start = bExt.genIntCast(256, false, adaptor.getStart());
    Value end = bExt.genIntCast(256, false, adaptor.getEnd());

    Value dataAddr = evmB.genDataAddrPtr(arr, arrTy);
    Value length = evmB.genDynSize(arr, arrTy);

    // Validate: start <= end
    auto startGtEnd =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, start, end);
    evmB.genRevertWithMsg(startGtEnd, "Slice starts after end", loc);

    // Validate: end <= length
    auto endGtLen =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, end, length);
    evmB.genRevertWithMsg(endGtLen, "Slice is greater than length", loc);

    // Compute stride (element size in calldata)
    unsigned stride = 32;
    if (auto arrType = dyn_cast<sol::ArrayType>(arrTy))
      stride = evm::getCallDataHeadSize(arrType.getEltType());
    else if (isa<sol::StringType>(arrTy))
      stride = 1;

    // newOffset = dataAddr + start * stride
    Value scaledStart =
        r.create<arith::MulIOp>(loc, start, bExt.genI256Const(stride));
    Value newOffset = r.create<arith::AddIOp>(loc, dataAddr, scaledStart);

    // newLength = end - start
    Value newLength = r.create<arith::SubIOp>(loc, end, start);

    r.replaceOp(op, bExt.genLLVMStruct({newOffset, newLength}));
    return success();
  }
};

struct CopyOpLowering : public OpConversionPattern<sol::CopyOp> {
  using OpConversionPattern<sol::CopyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);

    evmB.genCopy(srcTy, adaptor.getSrc(), adaptor.getDst(), srcDataLoc,
                 dstDataLoc, loc);
    r.eraseOp(op);
    return success();
  }
};

struct ThisOpLowering : public OpRewritePattern<sol::ThisOp> {
  using OpRewritePattern<sol::ThisOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ThisOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::AddressOp>(op);
    return success();
  }
};

struct CallerOpLowering : public OpRewritePattern<sol::CallerOp> {
  using OpRewritePattern<sol::CallerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallerOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::CallerOp>(op);
    return success();
  }
};

struct OriginOpLowering : public OpRewritePattern<sol::OriginOp> {
  using OpRewritePattern<sol::OriginOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::OriginOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::OriginOp>(op);
    return success();
  }
};

struct GasPriceOpLowering : public OpRewritePattern<sol::GasPriceOp> {
  using OpRewritePattern<sol::GasPriceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::GasPriceOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::GasPriceOp>(op);
    return success();
  }
};

struct CallValueOpLowering : public OpRewritePattern<sol::CallValueOp> {
  using OpRewritePattern<sol::CallValueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::CallValueOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::CallValOp>(op);
    return success();
  }
};

struct SigOpLowering : public OpConversionPattern<sol::SigOp> {
  using OpConversionPattern<sol::SigOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::SigOp op, OpAdaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    Value zero = bExt.genI256Const(0);
    Value data = r.create<yul::CallDataLoadOp>(loc, zero);
    // Mask to keep only the top 4 bytes (bytes4 is MSB-aligned).
    Value mask = bExt.genI256Const(llvm::APInt::getHighBitsSet(256, 32));
    r.replaceOp(op, r.create<arith::AndIOp>(loc, data, mask));
    return success();
  }
};

struct BaseFeeOpLowering : public OpRewritePattern<sol::BaseFeeOp> {
  using OpRewritePattern<sol::BaseFeeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::BaseFeeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::BaseFeeOp>(op);
    return success();
  }
};

struct BlobBaseFeeOpLowering : public OpRewritePattern<sol::BlobBaseFeeOp> {
  using OpRewritePattern<sol::BlobBaseFeeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::BlobBaseFeeOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::BlobBaseFeeOp>(op);
    return success();
  }
};

struct ChainIdOpLowering : public OpRewritePattern<sol::ChainIdOp> {
  using OpRewritePattern<sol::ChainIdOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::ChainIdOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::ChainIdOp>(op);
    return success();
  }
};

struct CoinbaseOpLowering : public OpRewritePattern<sol::CoinbaseOp> {
  using OpRewritePattern<sol::CoinbaseOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::CoinbaseOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::CoinBaseOp>(op);
    return success();
  }
};

struct DifficultyOpLowering : public OpRewritePattern<sol::DifficultyOp> {
  using OpRewritePattern<sol::DifficultyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::DifficultyOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::PrevrandaoOp>(op);
    return success();
  }
};

struct GasLimitOpLowering : public OpRewritePattern<sol::GasLimitOp> {
  using OpRewritePattern<sol::GasLimitOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::GasLimitOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::GasLimitOp>(op);
    return success();
  }
};

struct BlockNumberOpLowering : public OpRewritePattern<sol::BlockNumberOp> {
  using OpRewritePattern<sol::BlockNumberOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::BlockNumberOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::NumberOp>(op);
    return success();
  }
};

struct PrevRandaoOpLowering : public OpRewritePattern<sol::PrevRandaoOp> {
  using OpRewritePattern<sol::PrevRandaoOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::PrevRandaoOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::PrevrandaoOp>(op);
    return success();
  }
};

struct TimestampOpLowering : public OpRewritePattern<sol::TimestampOp> {
  using OpRewritePattern<sol::TimestampOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::TimestampOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::TimeStampOp>(op);
    return success();
  }
};

struct GasLeftOpLowering : public OpRewritePattern<sol::GasLeftOp> {
  using OpRewritePattern<sol::GasLeftOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sol::GasLeftOp op,
                                PatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::GasOp>(op);
    return success();
  }
};

struct LibAddrOpLowering : public OpConversionPattern<sol::LibAddrOp> {
  using OpConversionPattern<sol::LibAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LibAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::LinkerSymbolOp>(op, op.getName());
    return success();
  }
};

struct CodeHashOpLowering : public OpConversionPattern<sol::CodeHashOp> {
  using OpConversionPattern<sol::CodeHashOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CodeHashOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::ExtCodeHashOp>(op, adaptor.getContAddr());
    return success();
  }
};

struct BalanceOpLowering : public OpConversionPattern<sol::BalanceOp> {
  using OpConversionPattern<sol::BalanceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BalanceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::BalanceOp>(op, adaptor.getContAddr());
    return success();
  }
};

struct EncodeOpLowering : public OpConversionPattern<sol::EncodeOp> {
  using OpConversionPattern<sol::EncodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EncodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value freePtr = evmB.genFreePtr();
    if (op.getPacked()) {
      Value dataStart =
          r.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(32));
      Value dataEnd = evmB.genABIPackedEncoding(op.getIns().getType(),
                                                adaptor.getIns(), dataStart);
      Value dataSize = r.create<arith::SubIOp>(loc, dataEnd, dataStart);
      r.create<yul::MStoreOp>(loc, freePtr, dataSize);
      Value allocationSize = r.create<arith::SubIOp>(loc, dataEnd, freePtr);
      evmB.genFreePtrUpd(freePtr, allocationSize);
    } else if (op.getSelector()) {
      assert(adaptor.getSelector() && "selector operand is required");

      Value selectorAddr =
          r.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(32));
      r.create<yul::MStoreOp>(loc, selectorAddr, adaptor.getSelector());

      Value tupleStart =
          r.create<arith::AddIOp>(loc, selectorAddr, bExt.genI256Const(4));
      Value tupleEnd = evmB.genABITupleEncoding(op.getIns().getType(),
                                                adaptor.getIns(), tupleStart);
      Value dataSize = r.create<arith::SubIOp>(loc, tupleEnd, selectorAddr);
      r.create<yul::MStoreOp>(loc, freePtr, dataSize);
      Value allocationSize = r.create<arith::SubIOp>(loc, tupleEnd, freePtr);
      evmB.genFreePtrUpd(freePtr, allocationSize);
    } else {
      Value tupleStart =
          r.create<arith::AddIOp>(loc, freePtr, bExt.genI256Const(32));
      Value tupleEnd = evmB.genABITupleEncoding(op.getIns().getType(),
                                                adaptor.getIns(), tupleStart);
      Value tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);
      r.create<yul::MStoreOp>(loc, freePtr, tupleSize);
      Value allocationSize = r.create<arith::SubIOp>(loc, tupleEnd, freePtr);
      evmB.genFreePtrUpd(freePtr, allocationSize);
    }
    r.replaceOp(op, freePtr);
    return success();
  }
};

struct DecodeOpLowering : public OpConversionPattern<sol::DecodeOp> {
  using OpConversionPattern<sol::DecodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DecodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    std::vector<Value> results;
    Value tupleSize = r.create<yul::MLoadOp>(loc, adaptor.getAddr());
    Value tupleStart =
        r.create<arith::AddIOp>(loc, adaptor.getAddr(), bExt.genI256Const(32));
    Value tupleEnd = r.create<arith::AddIOp>(loc, tupleStart, tupleSize);
    bool fromMem = sol::getDataLocation(op.getAddr().getType()) ==
                   sol::DataLocation::Memory;
    assert(fromMem && "NYI");
    evmB.genABITupleDecoding(op.getResultTypes(), tupleStart, tupleEnd, results,
                             fromMem);
    r.replaceOp(op, results);
    return success();
  }
};

struct ExtCallOpLowering : public OpConversionPattern<sol::ExtCallOp> {
  using OpConversionPattern<sol::ExtCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);
    auto mod = op->getParentOfType<ModuleOp>();

    assert(sol::evmCanOverchargeGasForCall(mod) && "NYI");

    // TODO:
    // - The return arg analysis is done for evm's the supports return data (See
    // solidity::frontend::ReturnInfo).
    // - The generated code for the return has returndatacopy and returnsize.
    assert(sol::evmSupportsReturnData(mod) && "NYI");

    // Check if we need to generate the extcodesize check.
    unsigned totHeadSize = 0;
    for (auto resTy : op.getCalleeType().getResults()) {
      totHeadSize += evm::getCallDataHeadSize(resTy);
    }
    // TODO: Do we really need to check revertStrings() >= RevertStrings::Debug
    // here?
    bool extCodeSizeCheck =
        totHeadSize == 0 || !sol::evmSupportsReturnData(mod);

    if (extCodeSizeCheck) {
      // Generate the revert code.
      auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, adaptor.getAddr());
      auto isExtCodeSizeZero = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, extCodeSize, bExt.genI256Const(0));
      if (sol::isRevertStringsEnabled(mod))
        evmB.genRevertWithMsg(isExtCodeSizeZero,
                              "Target contract does not contain code");
      else
        evmB.genRevert(isExtCodeSizeZero);
    }

    // Generate the store of the selector.
    Value selectorAddr = evmB.genFreePtr();
    auto shiftedSelector = APInt(/*numBits=*/256, op.getSelector()) << 224;
    r.create<yul::MStoreOp>(loc, selectorAddr,
                            bExt.genI256Const(shiftedSelector));

    // Generate the abi encoding code.
    Value tupleStart =
        r.create<arith::AddIOp>(loc, selectorAddr, bExt.genI256Const(4));
    Value tupleEnd = evmB.genABITupleEncoding(op.getIns().getType(),
                                              adaptor.getIns(), tupleStart);

    // Calculate the return size statically and/or check if it's dynamic. This
    // is copied from solidity::frontend::ReturnInfo.
    unsigned staticRetSizeVal = 0;
    bool isRetSizeDynamic = false;
    for (Type ty : op.getCalleeType().getResults()) {
      if (sol::isDynamicallySized(ty)) {
        isRetSizeDynamic = true;
        staticRetSizeVal = 0;
        break;
      }
      staticRetSizeVal += evm::getCallDataHeadSize(ty);
    }

    // Generate the call.
    Value inpSize = r.create<arith::SubIOp>(loc, tupleEnd, selectorAddr);
    Value staticRetSize = bExt.genI256Const(staticRetSizeVal);
    mlir::Value status;

    // Order is important here, staticcall might overlap with delegatecall.
    if (op.getDelegateCall())
      status = r.create<yul::DelegateCallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);
    else if (op.getStaticCall())
      status = r.create<yul::StaticCallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);
    else
      status = r.create<yul::CallOp>(
          loc, adaptor.getGas(), adaptor.getAddr(), adaptor.getVal(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);

    // Generate forwarding revert if not try-call.
    if (!op.getTryCall()) {
      auto statusIsZero = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  status, bExt.genI256Const(0));
      evmB.genForwardingRevert(statusIsZero);
    }

    // Get the types of the results from the decoding, which should be the same
    // as the corsp legal types.
    SmallVector<Type> decodedResultTys;
    if (failed(getTypeConverter()->convertTypes(op.getCalleeType().getResults(),
                                                decodedResultTys)))
      return failure();

    // Generate the if-else op of the status that yields the decoded results.
    auto statusIsNotZero = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, status, bExt.genI256Const(0));
    auto statusIfOp = r.create<scf::IfOp>(loc, /*resultTys=*/decodedResultTys,
                                          /*cond=*/statusIsNotZero);

    // Generate the else block (failure) which yields undefs. The undefs will
    // not be used during execution as the either control flow will skip the
    // sol.try's success block or hit the previous forwarding revert.
    r.setInsertionPointToStart(&statusIfOp.getElseRegion().emplaceBlock());
    SmallVector<Value, 2> undefYields;
    for (Type ty : decodedResultTys) {
      undefYields.push_back(r.create<LLVM::UndefOp>(loc, ty));
    }
    r.create<scf::YieldOp>(loc, undefYields);

    // Generte the then block (success).
    r.setInsertionPointToStart(&statusIfOp.getThenRegion().emplaceBlock());

    // The allocation `selectorAddr` will be reused for the return data.

    // Generate the decoding of the results.
    Value retDataSize = r.create<yul::ReturnDataSizeOp>(loc);
    std::vector<Value> decodedResults;
    if (isRetSizeDynamic) {
      r.create<yul::ReturnDataCopyOp>(loc, selectorAddr,
                                      /*src=*/bExt.genI256Const(0),
                                      retDataSize);
      evmB.genFreePtrUpd(selectorAddr, retDataSize);
      Value tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, retDataSize);
      evmB.genABITupleDecoding(op.getCalleeType().getResults(), selectorAddr,
                               tupleEnd, decodedResults, /*fromMem=*/true);
    } else {
      // See https://github.com/ethereum/solidity/pull/12684
      Value staticRetSizeGreater = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ugt, staticRetSize, retDataSize);

      auto ifOp = r.create<scf::IfOp>(loc, /*resultTy=*/r.getIntegerType(256),
                                      /*cond=*/staticRetSizeGreater,
                                      /*withElse=*/true);
      // Then block:
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      evmB.genFreePtrUpd(selectorAddr, retDataSize);
      Value tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, retDataSize);
      r.create<scf::YieldOp>(loc, tupleEnd);
      // Else block:
      r.setInsertionPointToStart(&ifOp.getElseRegion().front());
      evmB.genFreePtrUpd(selectorAddr, staticRetSize);
      tupleEnd = r.create<arith::AddIOp>(loc, selectorAddr, staticRetSize);
      r.create<scf::YieldOp>(loc, tupleEnd);

      r.setInsertionPointAfter(ifOp);
      evmB.genABITupleDecoding(op.getCalleeType().getResults(), selectorAddr,
                               ifOp.getResult(0), decodedResults,
                               /*fromMem=*/true);
    }
    r.create<scf::YieldOp>(loc, decodedResults);

    // Replace the sol.ext_call op with the status check + the decoded results.
    assert(decodedResults.size() <= 1 && "NYI");
    SmallVector<Value, 2> newResults{statusIsNotZero};
    newResults.append(statusIfOp.getResults().begin(),
                      statusIfOp.getResults().end());
    r.replaceOp(op, newResults);
    return success();
  }
};

template <typename OpT, typename AdaptorT, typename OpBuilderFuncT>
static LogicalResult lowerBareCallLikeOp(OpT op, AdaptorT adaptor,
                                         ConversionPatternRewriter &r,
                                         OpBuilderFuncT &&opBuilderFunc) {
  Location loc = op.getLoc();
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(r, loc);

  auto mod = op->template getParentOfType<ModuleOp>();
  assert(sol::evmSupportsReturnData(mod) && "NYI");

  auto inpTy = cast<sol::StringType>(op.getInp().getType());
  Value inpSize = evmB.genDynSize(adaptor.getInp(), inpTy, loc);
  Value inpStart = evmB.genDataAddrPtr(adaptor.getInp(), inpTy, loc);
  Value zero = bExt.genI256Const(0);
  Value rawStatus = opBuilderFunc(loc, adaptor, inpStart, inpSize, zero, r);

  Value status = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                         rawStatus, bExt.genI256Const(0));
  Value retDataSize = r.create<yul::ReturnDataSizeOp>(loc);
  Value retData = evmB.genMemAllocForDynArray(
      retDataSize, bExt.genRoundUpToMultiple<32>(retDataSize), loc);
  Value retDataStart =
      evmB.genDataAddrPtr(retData, sol::DataLocation::Memory, loc);
  r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/retDataStart,
                                  /*src=*/bExt.genI256Const(0), retDataSize);

  r.replaceOp(op, SmallVector<Value, 2>{status, retData});
  return success();
}

struct BareCallOpLowering : public OpConversionPattern<sol::BareCallOp> {
  using OpConversionPattern<sol::BareCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BareCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    return lowerBareCallLikeOp(
        op, adaptor, r,
        [](Location loc, auto adaptor, Value inpStart, Value inpSize,
           Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::CallOp>(
              loc, adaptor.getGas(), adaptor.getAddr(), adaptor.getVal(),
              /*inpOffset=*/inpStart, inpSize, /*outOffset=*/zero,
              /*outSize=*/zero);
        });
  }
};

struct BareDelegateCallOpLowering
    : public OpConversionPattern<sol::BareDelegateCallOp> {
  using OpConversionPattern<sol::BareDelegateCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BareDelegateCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    return lowerBareCallLikeOp(
        op, adaptor, r,
        [](Location loc, auto adaptor, Value inpStart, Value inpSize,
           Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::DelegateCallOp>(
              loc, adaptor.getGas(), adaptor.getAddr(),
              /*inpOffset=*/inpStart, inpSize, /*outOffset=*/zero,
              /*outSize=*/zero);
        });
  }
};

struct BareStaticCallOpLowering
    : public OpConversionPattern<sol::BareStaticCallOp> {
  using OpConversionPattern<sol::BareStaticCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BareStaticCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    return lowerBareCallLikeOp(
        op, adaptor, r,
        [](Location loc, auto adaptor, Value inpStart, Value inpSize,
           Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::StaticCallOp>(
              loc, adaptor.getGas(), adaptor.getAddr(),
              /*inpOffset=*/inpStart, inpSize, /*outOffset=*/zero,
              /*outSize=*/zero);
        });
  }
};

/// Lowers the common value-transfer call pattern used by 'sol.send' and
/// 'sol.transfer' and returns the requested status predicate.
template <typename AdaptorT>
static Value genValueTransferStatus(Location loc, AdaptorT adaptor,
                                    arith::CmpIPredicate pred,
                                    ConversionPatternRewriter &r) {
  mlir::solgen::BuilderExt bExt(r, loc);
  Value zero = bExt.genI256Const(0);
  Value callStipend = bExt.genI256Const(2300);
  Value valueIsZero = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              adaptor.getVal(), zero);
  Value gas = r.create<arith::SelectOp>(loc, valueIsZero, callStipend, zero);

  Value callStatus = r.create<yul::CallOp>(
      loc, gas, adaptor.getAddr(), adaptor.getVal(),
      /*inpOffset=*/zero, /*inpSize=*/zero, /*outOffset=*/zero,
      /*outSize=*/zero);
  return r.create<arith::CmpIOp>(loc, pred, callStatus, zero);
}

struct SendOpLowering : public OpConversionPattern<sol::SendOp> {
  using OpConversionPattern<sol::SendOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::SendOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    Value status =
        genValueTransferStatus(loc, adaptor, arith::CmpIPredicate::ne, r);
    r.replaceOp(op, status);
    return success();
  }
};

struct TransferOpLowering : public OpConversionPattern<sol::TransferOp> {
  using OpConversionPattern<sol::TransferOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::TransferOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);

    Value statusIsZero =
        genValueTransferStatus(loc, adaptor, arith::CmpIPredicate::eq, r);
    evmB.genForwardingRevert(statusIsZero);
    r.eraseOp(op);
    return success();
  }
};

struct NewOpLowering : public OpConversionPattern<sol::NewOp> {
  using OpConversionPattern<sol::NewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::NewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value bytecodeAddr = evmB.genFreePtr();
    Value dataOffset = r.create<yul::DataOffsetOp>(loc, op.getObjName());
    Value dataSize = r.create<yul::DataSizeOp>(loc, op.getObjName());
    r.create<yul::CodeCopyOp>(loc, bytecodeAddr, dataOffset, dataSize);

    Value tupleStart = r.create<arith::AddIOp>(loc, bytecodeAddr, dataSize);
    Value tupleEnd = evmB.genABITupleEncoding(
        op.getCtorArgs().getType(), adaptor.getCtorArgs(), tupleStart);
    Value allocSize = r.create<arith::SubIOp>(loc, tupleEnd, bytecodeAddr);

    Value status;
    if (op.getSalt())
      status = r.create<yul::Create2Op>(loc, adaptor.getVal(), bytecodeAddr,
                                        allocSize, adaptor.getSalt());
    else
      status = r.create<yul::CreateOp>(loc, adaptor.getVal(), bytecodeAddr,
                                       allocSize);

    Value zero = bExt.genI256Const(0);
    Value statusIsZero =
        r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, status, zero);
    evmB.genForwardingRevert(statusIsZero);
    r.replaceOp(op, status);
    return success();
  }
};

struct ObjectCodeOpLowering : public OpConversionPattern<sol::ObjectCodeOp> {
  using OpConversionPattern<sol::ObjectCodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ObjectCodeOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);

    Value dataSize = r.create<yul::DataSizeOp>(loc, op.getObjName());
    Value alloc = evmB.genMemAlloc(op.getType(), /*zeroInit=*/false,
                                   /*initVals=*/{}, dataSize);
    Value dataAddr = evmB.genDataAddrPtr(alloc, sol::DataLocation::Memory);
    Value dataOffset = r.create<yul::DataOffsetOp>(loc, op.getObjName());
    r.create<yul::CodeCopyOp>(loc, dataAddr, dataOffset, dataSize);
    r.replaceOp(op, alloc);
    return success();
  }
};

struct CodeOpLowering : public OpConversionPattern<sol::CodeOp> {
  using OpConversionPattern<sol::CodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {

    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, adaptor.getContAddr());
    Value alloc = evmB.genMemAlloc(op.getType(), /*zeroInit=*/false,
                                   /*initVals=*/{}, extCodeSize);
    auto codeAddr = evmB.genDataAddrPtr(alloc, sol::DataLocation::Memory);
    r.create<yul::ExtCodeCopyOp>(
        loc, adaptor.getContAddr(), /*dstOffset=*/codeAddr,
        /*srcOffset=*/bExt.genI256Const(0), extCodeSize);
    r.replaceOp(op, alloc);
    return success();
  }
};

struct TryOpLowering : public OpConversionPattern<sol::TryOp> {
  using OpConversionPattern<sol::TryOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::TryOp tryOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = tryOp.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    auto ifStatus = r.create<sol::IfOp>(loc, tryOp.getStatus());

    //
    // Success region
    //

    if (tryOp.getSuccessRegion().empty()) {
      r.setInsertionPointToStart(&tryOp.getSuccessRegion().emplaceBlock());
      r.create<sol::YieldOp>(loc);
    } else {
      r.inlineRegionBefore(tryOp.getSuccessRegion(), ifStatus.getThenRegion(),
                           ifStatus.getThenRegion().begin());
    }

    //
    // Failure region
    //

    if (tryOp.getFallbackRegion().empty()) {
      evm::Builder evmB(r, loc);
      r.setInsertionPointToStart(&ifStatus.getElseRegion().emplaceBlock());
      evmB.genForwardingRevert();
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));
    } else {
      r.inlineRegionBefore(tryOp.getFallbackRegion(), ifStatus.getElseRegion(),
                           ifStatus.getElseRegion().begin());
    }

    if (tryOp.getPanicRegion().empty() && tryOp.getErrorRegion().empty()) {
      r.eraseOp(tryOp);
      return success();
    }

    r.setInsertionPointToStart(&ifStatus.getElseRegion().front());

    // Generate a flag to check if we need to run the fallback. The flag will be
    // set to false by any of the other clause.
    auto boolAllocaTy = sol::PointerType::get(r.getContext(), r.getI1Type(),
                                              sol::DataLocation::Stack);
    auto runFallbackFlag = r.create<sol::AllocaOp>(loc, boolAllocaTy);
    r.create<sol::StoreOp>(loc, bExt.genBool(true), runFallbackFlag);

    // Generate an if op that checks if the returndata is large enough to hold
    // the selector.
    //
    // The ops after this if op belong to the fallback. Track it for the
    // fallback lowering.
    auto returnDataSize = r.create<yul::ReturnDataSizeOp>(loc);
    auto selectorRetCond = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ugt, returnDataSize, bExt.genI256Const(3));
    auto ifSelectorRet = r.create<sol::IfOp>(loc, selectorRetCond);
    auto fallbackPoint = r.saveInsertionPoint().getPoint();

    //
    // Selector region
    //
    r.setInsertionPointToStart(&ifSelectorRet.getThenRegion().emplaceBlock());
    r.setInsertionPoint(r.create<sol::YieldOp>(loc));

    // Generate the selector extraction code.
    auto zero = bExt.genI256Const(0);
    r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/zero, /*src=*/zero,
                                    /*size=*/bExt.genI256Const(4));
    auto selectorWord = r.create<yul::MLoadOp>(loc, zero);
    auto selector =
        r.create<arith::ShRUIOp>(loc, selectorWord, bExt.genI256Const(224));

    // Generate the switch for the panic and error catch blocks.
    SmallVector<APInt, 2> switchSelectors;
    APInt panicSelector(
        /*numBits=*/256, mlir::evm::selectorFromSignature("Panic(uint256)"));
    APInt errorSelector(
        /*numBits=*/256, mlir::evm::selectorFromSignature("Error(string)"));
    if (!tryOp.getPanicRegion().empty())
      switchSelectors.push_back(panicSelector);
    if (!tryOp.getErrorRegion().empty())
      switchSelectors.push_back(errorSelector);
    assert(!switchSelectors.empty());
    auto switchSelectorsAttr = mlir::DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(switchSelectors.size()),
                              r.getIntegerType(256)),
        switchSelectors);
    auto switchOp = r.create<sol::SwitchOp>(loc, selector, switchSelectorsAttr,
                                            switchSelectors.size());
    r.setInsertionPointToStart(&switchOp.getDefaultRegion().emplaceBlock());
    r.setInsertionPoint(r.create<sol::YieldOp>(loc));

    //
    // Panic case region
    //
    if (!tryOp.getPanicRegion().empty()) {
      // FIXME: We should query for the case region using the attribute!
      r.setInsertionPointToStart(&switchOp.getCaseRegions()[0].emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));

      // Genereate an if op that checks if the returndata is large enough to
      // hold the panic code.
      auto panicRetCond =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                  returnDataSize, bExt.genI256Const(0x23));
      auto ifPanicRet = r.create<sol::IfOp>(loc, panicRetCond);

      // Inline the panic region.
      r.inlineRegionBefore(tryOp.getPanicRegion(), ifPanicRet.getThenRegion(),
                           ifPanicRet.getThenRegion().begin());
      Block &thenEntry = ifPanicRet.getThenRegion().front();
      r.setInsertionPointToStart(&thenEntry);
      r.create<sol::StoreOp>(loc, bExt.genBool(false), runFallbackFlag);
      // Replace the panic code block arg with the panic code at offset 4.
      BlockArgument blkArg = thenEntry.getArgument(0);
      r.create<yul::ReturnDataCopyOp>(blkArg.getLoc(), /*dst=*/zero,
                                      /*src=*/bExt.genI256Const(4),
                                      /*size=*/bExt.genI256Const(0x20));
      Value panicCode = r.create<yul::MLoadOp>(blkArg.getLoc(), zero);
      auto blkArgRepl = getTypeConverter()->materializeSourceConversion(
          r, loc, blkArg.getType(), panicCode);
      // FIXME: Why does the following cause a "no matched legalization pattern"
      // of the op from the materialization? Is this related to early
      // legalization of in-place modifications?
      // (https://discourse.llvm.org/t/dialect-conversion-fails-if-some-requires-recursive-application/79371/6)
      // r.replaceAllUsesWith(blkArg, blkArgRepl);
      blkArg.replaceAllUsesWith(blkArgRepl);
      thenEntry.eraseArgument(0);
    }

    //
    // Error case region
    //
    if (!tryOp.getErrorRegion().empty()) {
      // FIXME: We should query for the case region using the attribute!
      unsigned errorCaseIdx = switchSelectors.size() == 2 ? 1 : 0;
      r.setInsertionPointToStart(
          &switchOp.getCaseRegions()[errorCaseIdx].emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));

      // Genereate an if op that checks if the returndata is large enough to
      // hold the error message.
      auto errMsgRetCond =
          r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                  returnDataSize, bExt.genI256Const(0x44));
      auto ifErrMsg = r.create<sol::IfOp>(loc, errMsgRetCond);

      // Inline the error region.
      r.inlineRegionBefore(tryOp.getErrorRegion(), ifErrMsg.getThenRegion(),
                           ifErrMsg.getThenRegion().begin());
      Block &thenEntry = ifErrMsg.getThenRegion().front();
      r.setInsertionPointToStart(&thenEntry);
      r.create<sol::StoreOp>(loc, bExt.genBool(false), runFallbackFlag);

      // Generate the error message extraction code from the return data and
      // replace the block argument with it.
      //
      // TODO: Is it necessary to generate all the checks in
      // YulUtilFunctions::tryDecodeErrorMessageFunction()?
      BlockArgument blkArg = thenEntry.getArgument(0);
      Location loc = blkArg.getLoc();
      evm::Builder evmB(r, loc);
      Value abiTupleSize =
          r.create<arith::SubIOp>(loc, returnDataSize, bExt.genI256Const(4));
      Value abiTuple = evmB.genMemAlloc(abiTupleSize);
      r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/abiTuple,
                                      /*src=*/bExt.genI256Const(4),
                                      abiTupleSize);
      Value errMsgOffset = r.create<yul::MLoadOp>(loc, abiTuple);
      Value errMsg = r.create<arith::AddIOp>(loc, abiTuple, errMsgOffset);
      auto blkArgRepl = getTypeConverter()->materializeSourceConversion(
          r, loc, blkArg.getType(), errMsg);
      // FIXME: See panic clause lowering.
      blkArg.replaceAllUsesWith(blkArgRepl);
      thenEntry.eraseArgument(0);
    }

    //
    // Fallback region
    //
    Block *fallbackBlk = r.splitBlock(fallbackPoint->getBlock(), fallbackPoint);
    r.setInsertionPointAfter(ifSelectorRet);
    auto runFallbackFlagLd = r.create<sol::LoadOp>(loc, runFallbackFlag);
    auto ifRunFallback = r.create<sol::IfOp>(loc, runFallbackFlagLd);
    ifRunFallback.getThenRegion().emplaceBlock();
    r.inlineBlockBefore(fallbackBlk, &ifRunFallback.getThenRegion().front(),
                        ifRunFallback.getThenRegion().front().begin());
    r.create<sol::YieldOp>(loc);

    r.eraseOp(tryOp);
    return success();
  }
};

struct AssertOpLowering : public OpConversionPattern<sol::AssertOp> {
  using OpConversionPattern<sol::AssertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::AssertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(r, loc);

    // Generate: if (!cond) { panic(0x01) }
    mlir::Value falseVal =
        r.create<arith::ConstantIntOp>(loc, r.getI1Type(), 0);
    mlir::Value negCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  op.getCond(), falseVal);
    evmB.genPanic(evm::PanicCode::Assert, negCond);

    r.eraseOp(op);
    return success();
  }
};

struct RequireOpLowering : public OpConversionPattern<sol::RequireOp> {
  using OpConversionPattern<sol::RequireOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::RequireOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // Generate the revert condition.
    mlir::Value falseVal =
        r.create<arith::ConstantIntOp>(loc, r.getI1Type(), 0);
    mlir::Value negCond = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  op.getCond(), falseVal);
    if (op.getCall()) {
      assert(!op.getMsg().empty());
      evmB.genRevert(negCond, op.getArgs().getTypes(), adaptor.getArgs(),
                     op.getMsg());
      r.eraseOp(op);
      return success();
    }

    // Generate the revert.
    if (!op.getMsg().empty())
      evmB.genRevertWithMsg(negCond, op.getMsg().str());
    else
      evmB.genRevert(negCond);

    r.eraseOp(op);
    return success();
  }
};

struct EmitOpLowering : public OpConversionPattern<sol::EmitOp> {
  using OpConversionPattern<sol::EmitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EmitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // Collect the remapped indexed and non-indexed args.
    //
    // FIXME: How do we get `extraClassDeclaration` functions to be part of the
    // OpAdaptor?
    auto remappedOperands = adaptor.getOperands();
    std::vector<Value> indexedArgs, nonIndexedArgs;
    std::vector<Type> nonIndexedArgsType;
    if (op.getSignature()) {
      auto signatureHash = evm::keccak256AsAPInt(op.getSignature()->str());
      indexedArgs.push_back(bExt.genI256Const(signatureHash));
    }
    unsigned argIdx = 0;
    while (argIdx < op.getIndexedArgsCount())
      indexedArgs.push_back(remappedOperands[argIdx++]);
    for (Value arg : op.getNonIndexedArgs()) {
      nonIndexedArgsType.push_back(arg.getType());
      nonIndexedArgs.push_back(remappedOperands[argIdx++]);
    }

    // Generate the tuple encoding for the non-indexed args.
    // TODO: Are we sure we need an unbounded allocation here?
    Value tupleStart = evmB.genFreePtr();
    Value tupleEnd = evmB.genABITupleEncoding(nonIndexedArgsType,
                                              nonIndexedArgs, tupleStart);
    Value tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);

    // Generate sol.log and replace sol.emit with it.
    r.replaceOpWithNewOp<yul::LogOp>(op, tupleStart, tupleSize, indexedArgs);

    return success();
  }
};

struct RevertOpLowering : public OpConversionPattern<sol::RevertOp> {
  using OpConversionPattern<sol::RevertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::RevertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    evm::Builder evmB(r, op.getLoc());
    if (op.getCall()) {
      // revert ErrorName(...)
      assert(!op.getSignature().empty());
      evmB.genRevert(op.getArgs().getTypes(), adaptor.getArgs(),
                     op.getSignature());
    } else if (!op.getSignature().empty()) {
      // revert("reason")
      evmB.genRevertWithMsg(op.getSignature().str());
    } else {
      // revert()
      mlir::solgen::BuilderExt bExt(r, op.getLoc());
      mlir::Value zero = bExt.genI256Const(0);
      r.create<yul::RevertOp>(op.getLoc(), zero, zero);
    }
    r.eraseOp(op);
    return success();
  }
};

// (Copied and modified from clangir).
struct IfOpLowering : public OpRewritePattern<sol::IfOp> {
  using OpRewritePattern<sol::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::IfOp ifOp,
                                PatternRewriter &r) const override {
    Location loc = ifOp.getLoc();

    bool emptyElse = ifOp.getElseRegion().empty();
    Block *currentBlock = r.getInsertionBlock();
    Block *remainingOpsBlock =
        r.splitBlock(currentBlock, r.getInsertionPoint());
    Block *continueBlock;
    if (ifOp->getResults().empty())
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region.
    Block *thenBeforeBody = &ifOp.getThenRegion().front();
    Block *thenAfterBody = &ifOp.getThenRegion().back();
    r.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    r.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<sol::YieldOp>(thenAfterBody->getTerminator())) {
      r.replaceOpWithNewOp<cf::BranchOp>(thenYieldOp, thenYieldOp.getIns(),
                                         continueBlock);
    }

    r.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    Block *elseBeforeBody = nullptr;
    Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      r.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    r.setInsertionPointToEnd(currentBlock);
    r.create<cf::CondBranchOp>(loc, ifOp.getCond(), thenBeforeBody,
                               elseBeforeBody);

    if (!emptyElse) {
      r.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<sol::YieldOp>(elseAfterBody->getTerminator())) {
        r.replaceOpWithNewOp<cf::BranchOp>(elseYieldOp, elseYieldOp.getIns(),
                                           continueBlock);
      }
    }

    r.replaceOp(ifOp, continueBlock->getArguments());
    return success();
  }
};

struct SwitchOpLowering : public OpRewritePattern<sol::SwitchOp> {
  using OpRewritePattern<sol::SwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::SwitchOp switchOp,
                                PatternRewriter &r) const override {
    // Split the block at the op.
    Block *condBlk = r.getInsertionBlock();
    Block *continueBlk = r.splitBlock(condBlk, Block::iterator(switchOp));

    auto convertRegion = [&](Region &region) -> Block * {
      for (Block &blk : region) {
        auto yield = dyn_cast<sol::YieldOp>(blk.getTerminator());
        if (!yield)
          continue;
        // Convert the yield terminator to a branch to the continue block.
        r.setInsertionPoint(yield);
        r.replaceOpWithNewOp<cf::BranchOp>(yield, continueBlk,
                                           yield.getOperands());
      }

      // Inline the region.
      Block *entryBlk = &region.front();
      r.inlineRegionBefore(region, continueBlk);
      return entryBlk;
    };

    // Convert the case regions.
    SmallVector<Block *> caseSuccessors;
    SmallVector<llvm::APInt> caseVals;
    caseSuccessors.reserve(switchOp.getCases().size());
    caseVals.reserve(switchOp.getCases().size());
    for (auto [region, val] :
         llvm::zip(switchOp.getCaseRegions(), switchOp.getCases())) {
      caseSuccessors.push_back(convertRegion(region));
      caseVals.push_back(val);
    }

    // Convert the default region.
    Block *defaultBlk = convertRegion(switchOp.getDefaultRegion());

    if (caseVals.empty()) {
      r.replaceOp(switchOp, continueBlk->getArguments());
      return success();
    }

    // Create the switch.
    r.setInsertionPointToEnd(condBlk);
    SmallVector<ValueRange> caseOperands(caseSuccessors.size(), {});

    // Create the attribute for the case values.
    auto caseValsAttr = DenseIntElementsAttr::get(
        VectorType::get(static_cast<int64_t>(caseVals.size()),
                        switchOp.getArg().getType()),
        caseVals);

    r.create<cf::SwitchOp>(switchOp.getLoc(), switchOp.getArg(), defaultBlk,
                           ValueRange(), caseValsAttr, caseSuccessors,
                           caseOperands);
    r.replaceOp(switchOp, continueBlk->getArguments());
    return success();
  }
};

// (Copied and modified from clangir).
struct LoopOpInterfaceLowering
    : public OpInterfaceRewritePattern<sol::LoopOpInterface> {
  using OpInterfaceRewritePattern<
      sol::LoopOpInterface>::OpInterfaceRewritePattern;

  /// Walks a region while skipping operations of type `Ops`. This ensures the
  /// callback is not applied to said operations and its children.
  template <typename... Ops>
  void
  walkRegionSkipping(Region &region,
                     function_ref<WalkResult(Operation *)> callback) const {
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<Ops...>(op))
        return WalkResult::skip();
      return callback(op);
    });
  }

  /// Lowers operations with the terminator trait that have a single successor.
  void lowerTerminator(Operation *op, Block *dest, PatternRewriter &r) const {
    assert(op->hasTrait<OpTrait::IsTerminator>() && "not a terminator");
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPoint(op);
    r.replaceOpWithNewOp<cf::BranchOp>(op, dest);
  }

  void lowerConditionOp(sol::ConditionOp op, Block *body, Block *exit,
                        PatternRewriter &r) const {
    OpBuilder::InsertionGuard guard(r);
    r.setInsertionPoint(op);
    r.replaceOpWithNewOp<cf::CondBranchOp>(op, op.getCondition(), body, exit);
  }

  LogicalResult matchAndRewrite(sol::LoopOpInterface op,
                                PatternRewriter &r) const override {
    // Setup CFG blocks.
    Block *entry = r.getInsertionBlock();
    Block *exit = r.splitBlock(entry, r.getInsertionPoint());
    Block *cond = &op.getCond().front();
    Block *body = &op.getBody().front();
    Block *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    r.setInsertionPointToEnd(entry);
    r.create<cf::BranchOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<sol::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, r);

    // TODO: Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialectConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](Operation *op) {
      if (!isa<sol::ContinueOp>(op))
        return WalkResult::advance();

      lowerTerminator(op, dest, r);
      return WalkResult::skip();
    });

    // Lower break statements.
    // FIXME: Skip sol.switch once we implement it.
    walkRegionSkipping<sol::LoopOpInterface /* TODO:, sol::SwitchOp */>(
        op.getBody(), [&](Operation *op) {
          if (!isa<sol::BreakOp>(op))
            return WalkResult::advance();

          lowerTerminator(op, exit, r);
          return mlir::WalkResult::skip();
        });

    // Lower optional body region yield.
    for (Block &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<sol::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), r);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<sol::YieldOp>(step->getTerminator()), cond, r);

    // Move region contents out of the loop op.
    r.inlineRegionBefore(op.getCond(), exit);
    r.inlineRegionBefore(op.getBody(), exit);
    if (step)
      r.inlineRegionBefore(*op.maybeGetStep(), exit);

    r.eraseOp(op);
    return success();
  }
};

struct CallOpLowering : public OpConversionPattern<sol::CallOp> {
  using OpConversionPattern<sol::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    SmallVector<Type> convertedResTys;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                convertedResTys)))
      return failure();
    r.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), convertedResTys,
                                       adaptor.getOperands());
    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<sol::ReturnOp> {
  using OpConversionPattern<sol::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<sol::FuncOp> {
  using OpConversionPattern<sol::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // Set llvm.linkage attribute to private if not explicitly specified.
    std::vector<NamedAttribute> attrs;
    if (auto linkageAttr = op->getAttr("llvm.linkage"))
      attrs.push_back(r.getNamedAttr("llvm.linkage", linkageAttr));
    else
      attrs.push_back(r.getNamedAttr(
          "llvm.linkage",
          LLVM::LinkageAttr::get(r.getContext(), LLVM::Linkage::Private)));

    // Add the nofree and null_pointer_is_valid attributes of llvm via the
    // passthrough attribute.
    std::vector<Attribute> passthroughAttrs;
    passthroughAttrs.push_back(r.getStringAttr("nofree"));
    passthroughAttrs.push_back(r.getStringAttr("null_pointer_is_valid"));
    attrs.push_back(r.getNamedAttr(
        "passthrough", ArrayAttr::get(r.getContext(), passthroughAttrs)));

    // TODO: Add additional attribute for -O0 and -Oz

    auto convertedFuncTy = cast<FunctionType>(
        getTypeConverter()->convertType(op.getFunctionType()));
    // FIXME: The location of the block arguments are lost here!
    auto newOp = r.create<func::FuncOp>(op.getLoc(), op.getName(),
                                        convertedFuncTy, attrs);
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    r.eraseOp(op);
    return success();
  }
};

struct ContractOpLowering : public OpRewritePattern<sol::ContractOp> {
  const char *libAddrName = "library_deploy_address";

  using OpRewritePattern<sol::ContractOp>::OpRewritePattern;

  /// Generate the call value check.
  void genCallValChk(PatternRewriter &r, Location loc) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    auto callVal = r.create<yul::CallValOp>(loc);
    auto callValChk = r.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                              callVal, bExt.genI256Const(0));
    evmB.genRevert(callValChk);
  };

  /// Generate the free pointer initialization.
  void genFreePtrInit(PatternRewriter &r, Location loc,
                      size_t reservedMem = 0) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    mlir::Value freeMem = bExt.genI256Const(
        mlir::evm::MemoryLayout::generalPurposeMemoryStart + reservedMem);
    r.create<yul::MStoreOp>(loc, bExt.genI256Const(64), freeMem);
  };

  /// Generates the dispatch to interface functions.
  void genDispatch(sol::ContractOp contrOp,
                   SmallVector<uint32_t, 4> const &selectors,
                   SmallVector<sol::FuncOp, 4> &ifcFns,
                   PatternRewriter &r) const {
    Location loc = contrOp.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    Value notDelegateCallCond;
    if (contrOp.getKind() == sol::ContractKind::Library) {
      auto libAddr = r.create<yul::LoadImmutableOp>(loc, r.getIntegerType(256),
                                                    libAddrName);
      auto currAddr = r.create<yul::AddressOp>(loc);
      notDelegateCallCond = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, libAddr, currAddr);
    }

    // Do nothing if there are no interface functions.
    if (ifcFns.empty())
      return;

    // Generate `if iszero(lt(calldatasize(), 4))` and set the insertion point
    // to its then block.
    auto callDataSz = r.create<yul::CallDataSizeOp>(loc);
    auto callDataSzCmp = r.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, callDataSz, bExt.genI256Const(4));
    auto ifOp =
        r.create<scf::IfOp>(loc, callDataSzCmp, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard insertGuard(r);
    r.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load the selector from the calldata.
    auto callDataLd = r.create<yul::CallDataLoadOp>(loc, bExt.genI256Const(0));
    Value callDataSelector =
        r.create<arith::ShRUIOp>(loc, callDataLd, bExt.genI256Const(224));
    callDataSelector =
        r.create<arith::TruncIOp>(loc, r.getIntegerType(32), callDataSelector);

    auto selectorsAttr = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(static_cast<int64_t>(selectors.size()),
                                    r.getIntegerType(32)),
        selectors);

    // Generate the switch op.
    auto switchOp = r.create<mlir::scf::IntSwitchOp>(
        loc, /*resultTypes=*/TypeRange{}, callDataSelector, selectorsAttr,
        selectors.size());

    // Generate the default block.
    {
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(r.createBlock(&switchOp.getDefaultRegion()));
      r.create<scf::YieldOp>(loc);
    }

    for (auto [caseRegion, ifcFnOp] :
         llvm::zip(switchOp.getCaseRegions(), ifcFns)) {
      OpBuilder::InsertionGuard insertGuard(r);
      mlir::Block *caseBlk = r.createBlock(&caseRegion);
      r.setInsertionPointToStart(caseBlk);

      assert(ifcFnOp.getStateMutability());
      sol::StateMutability stateMutability = *ifcFnOp.getStateMutability();
      if (contrOp.getKind() == sol::ContractKind::Library) {
        assert(stateMutability != sol::StateMutability::Payable);
        if (stateMutability > sol::StateMutability::View) {
          assert(notDelegateCallCond);
          evmB.genRevert(notDelegateCallCond);
        }
      }

      if (contrOp.getKind() != sol::ContractKind::Library &&
          stateMutability != sol::StateMutability::Payable) {
        genCallValChk(r, loc);
      }

      // Decode the input parameters (if required).
      FunctionType origIfcFnTy = *ifcFnOp.getOrigFnType();
      std::vector<Value> decodedArgs;
      if (!origIfcFnTy.getInputs().empty()) {
        evmB.genABITupleDecoding(origIfcFnTy.getInputs(),
                                 /*tupleStart=*/bExt.genI256Const(4),
                                 /*tupleEnd=*/callDataSz, decodedArgs,
                                 /*fromMem=*/false);
      }

      // Generate the actual call.
      auto callOp = r.create<sol::CallOp>(loc, ifcFnOp, decodedArgs);

      // Encode the result using the ABI's tuple encoder.
      auto tupleStart = evmB.genFreePtr();
      mlir::Value tupleSize;
      if (!callOp.getResultTypes().empty()) {
        auto tupleEnd = evmB.genABITupleEncoding(
            origIfcFnTy.getResults(), callOp.getResults(), tupleStart);
        tupleSize = r.create<arith::SubIOp>(loc, tupleEnd, tupleStart);
      } else {
        tupleSize = bExt.genI256Const(0);
      }

      // Generate the return.
      assert(tupleSize);
      r.create<yul::ReturnOp>(loc, tupleStart, tupleSize);

      r.create<mlir::scf::YieldOp>(loc);
    }
  }

  /// Collects reachable function from `fn` in `reachableFns`.
  void getReachableFuncs(sol::FuncOp fn,
                         llvm::SetVector<sol::FuncOp> &reachableFns) const {
    reachableFns.insert(fn);
    fn.walk([&](Operation *op) {
      FlatSymbolRefAttr calleeSym;
      if (auto callOp = dyn_cast<sol::CallOp>(op))
        calleeSym = callOp.getCalleeAttr();
      else if (auto fnRef = dyn_cast<sol::FuncConstantOp>(op))
        calleeSym = fnRef.getSymAttr();
      else
        return;

      auto callee =
          SymbolTable::lookupNearestSymbolFrom<sol::FuncOp>(fn, calleeSym);
      if (reachableFns.contains(callee))
        return;
      getReachableFuncs(callee, reachableFns);
    });
  }

  LogicalResult matchAndRewrite(sol::ContractOp op,
                                PatternRewriter &r) const override {
    mlir::Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(r, loc);

    // Generate the creation and runtime ObjectOp.
    auto creationObj = r.create<yul::ObjectOp>(loc, op.getName());
    r.setInsertionPointToStart(&creationObj.getBody().front());
    auto runtimeObj =
        r.create<yul::ObjectOp>(loc, std::string(op.getName()) + "_deployed");

    SmallVector<sol::FuncOp, 4> ifcFns;
    sol::FuncOp ctor, receiveFn, fallbackFn;
    SmallVector<uint32_t, 4> selectors;
    size_t reservedMemSize = 0;
    SmallVector<sol::ImmutableOp, 2> immOps;

    // Track relevant types of functions (with selector, ctor, fallback,
    // receive) immutables (and reserved memory) etc; Remove state variables.
    for (Operation &i :
         llvm::make_early_inc_range(op.getBody()->getOperations())) {
      // Functions
      if (auto fn = dyn_cast<sol::FuncOp>(i)) {
        if (auto selector = fn.getSelector()) {
          selectors.push_back(*selector);
          ifcFns.push_back(fn);
        }
        auto fnKind = fn.getKind();
        if (fnKind) {
          switch (*fnKind) {
          case sol::FunctionKind::Constructor:
            assert(!ctor);
            ctor = fn;
            break;
          case sol::FunctionKind::Fallback:
            assert(!fallbackFn);
            fallbackFn = fn;
            break;
          case sol::FunctionKind::Receive:
            assert(!receiveFn);
            receiveFn = fn;
            break;
          }
        }

        // Immutables
      } else if (auto immOp = dyn_cast<sol::ImmutableOp>(i)) {
        reservedMemSize += evm::getCallDataHeadSize(immOp.getType());
        immOps.push_back(immOp);

        // State variables
      } else if (isa<sol::StateVarOp>(i)) {
        r.eraseOp(&i);

      } else {
        llvm_unreachable("NYI");
      }
    }

    if (ctor) {
      // Clone functions reachable from the ctor to creation and runtime
      // objects.
      llvm::SetVector<sol::FuncOp> reachableFns;
      getReachableFuncs(ctor, reachableFns);
      ctor->moveBefore(creationObj.getEntryBlock(),
                       creationObj.getEntryBlock()->begin());
      for (auto reachableFn : reachableFns) {
        if (reachableFn == ctor)
          continue;
        // Clone in the creation object and move the original to runtime object.
        r.clone(*reachableFn);
        reachableFn->moveBefore(runtimeObj.getEntryBlock(),
                                runtimeObj.getEntryBlock()->begin());
        reachableFn.setRuntimeAttr(r.getUnitAttr());
      }
    }

    for (auto fn :
         llvm::make_early_inc_range(op.getBody()->getOps<sol::FuncOp>())) {
      fn.setRuntimeAttr(r.getUnitAttr());
      fn->moveBefore(runtimeObj.getEntryBlock(),
                     runtimeObj.getEntryBlock()->begin());
    }

    //
    // Creation context
    //

    r.setInsertionPointToStart(creationObj.getEntryBlock());

    genFreePtrInit(r, loc, reservedMemSize);

    if (!ctor) {
      genCallValChk(r, loc);
    } else {
      assert(ctor.getStateMutability());
      if (*ctor.getStateMutability() != sol::StateMutability::Payable)
        genCallValChk(r, loc);
    }

    // Generate the call to constructor (if required).
    if (ctor && op.getKind() != sol::ContractKind::Library) {
      auto progSize = r.create<yul::DataSizeOp>(loc, creationObj.getName());
      auto codeSize = r.create<yul::CodeSizeOp>(loc);
      auto argSize = r.create<arith::SubIOp>(loc, codeSize, progSize);
      Value tupleStart = evmB.genMemAlloc(argSize);
      r.create<yul::CodeCopyOp>(loc, tupleStart, progSize, argSize);
      std::vector<Value> decodedArgs;
      FunctionType ctorFnTy = *ctor.getOrigFnType();
      if (!ctorFnTy.getInputs().empty()) {
        evmB.genABITupleDecoding(
            ctorFnTy.getInputs(), tupleStart,
            /*tupleEnd=*/r.create<arith::AddIOp>(loc, tupleStart, argSize),
            decodedArgs,
            /*fromMem=*/true);
      }
      r.create<sol::CallOp>(loc, ctor, decodedArgs);
    }

    // Generate the codecopy of the runtime object for the return from the
    // creation object.
    auto freePtr = r.create<yul::MLoadOp>(loc, bExt.genI256Const(64));
    auto runtimeObjSym = FlatSymbolRefAttr::get(runtimeObj);
    auto runtimeObjOffset = r.create<yul::DataOffsetOp>(loc, runtimeObjSym);
    auto runtimeObjSize = r.create<yul::DataSizeOp>(loc, runtimeObjSym);
    r.create<yul::CodeCopyOp>(loc, freePtr, runtimeObjOffset, runtimeObjSize);

    // Generate setimmutable of the library address (the runtime object uses
    // this for the delegate call check).
    if (op.getKind() == sol::ContractKind::Library)
      r.create<yul::SetImmutableOp>(loc, freePtr, libAddrName,
                                    r.create<yul::AddressOp>(loc));

    // Generate setimmutable's and remove all sol.immutable ops.
    for (sol::ImmutableOp immOp : llvm::make_early_inc_range(immOps)) {
      assert(immOp->getAttr("addr"));
      auto addr = bExt.genI256Const(
          cast<IntegerAttr>(immOp->getAttr("addr")).getValue());
      auto val = r.create<yul::MLoadOp>(loc, addr);
      r.create<yul::SetImmutableOp>(loc, freePtr, immOp.getName(), val);
      r.eraseOp(immOp);
    }

    // Generate the return for the creation context.
    r.create<yul::ReturnOp>(loc, freePtr, runtimeObjSize);

    //
    // Runtime context
    //

    r.setInsertionPointToStart(runtimeObj.getEntryBlock());

    // Generate the memory init.
    // TODO: Confirm if this should be the same as in the creation context.
    genFreePtrInit(r, loc);

    // Generate the dispatch to interface functions.
    genDispatch(op, selectors, ifcFns, r);

    // Generate receive function.
    if (receiveFn) {
      auto callDataSz = r.create<yul::CallDataSizeOp>(loc);
      auto callDataSzIsZero = r.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, callDataSz, bExt.genI256Const(0));
      auto ifOp =
          r.create<scf::IfOp>(loc, callDataSzIsZero, /*withElseRegion=*/false);
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      r.create<sol::CallOp>(loc, receiveFn, /*operands=*/ValueRange{});
      r.create<yul::StopOp>(loc);
    }

    // Generate fallback function.
    if (fallbackFn) {
      FunctionType fallbackFnTy = fallbackFn.getFunctionType();
      assert(fallbackFnTy.getNumInputs() == fallbackFnTy.getNumResults() &&
             "NYI");
      (void)fallbackFnTy;

      if (fallbackFn.getStateMutability() != sol::StateMutability::Payable) {
        genCallValChk(r, loc);
      }
      r.create<sol::CallOp>(loc, fallbackFn, /*operands=*/ValueRange{});
      r.create<yul::StopOp>(loc);

    } else {
      // TODO: Generate error message.
      r.create<yul::RevertOp>(loc, bExt.genI256Const(0), bExt.genI256Const(0));
    }

    // Relocate global constants into their corresponding Creation/Runtime
    // objects. StringLitOpLowering creates LLVM::GlobalOp at module scope and
    // LLVM::AddressOfOp inside the yul::ObjectOp. Moving the global into the
    // ObjectOp lets the EVM backend associate the CODECOPY data with the right
    // object's data section. ObjectOpLowering (a later pass) then lifts the
    // global back to module scope for final code generation.
    ModuleOp mod = runtimeObj->getParentOfType<ModuleOp>();
    mod->walk([&mod, &r](LLVM::AddressOfOp addrOf) {
      StringRef name = addrOf.getGlobalName();
      if (!name.starts_with("__data_in_code_"))
        return;

      auto obj = addrOf->getParentOfType<yul::ObjectOp>();
      assert(obj);
      auto gOp = SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
          mod, r.getStringAttr(name));
      assert(gOp);
      gOp->moveBefore(obj.getEntryBlock(), obj.getEntryBlock()->begin());
    });

    // TODO? Make sure op.getBody() is either empty or has only ops marked for
    // deletion.
    r.eraseOp(op);
    // TODO: Subobjects
    return success();
  }
};

} // namespace

// TODO: Assert that the type converter is compatible with the conversions. (We
// could only accept SolTypeConverter instead, but do we need need to be that
// strict?) (Also, can we do the assert at build time? If not, then only in
// debug builds?)

void evm::populateArithPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ConstantOpLowering, FuncConstantOpLowering,
           DefaultFuncConstantOpLowering>(pats.getContext());
  pats.add<CastOpLowering, AddressCastOpLowering, ContractCastOpLowering,
           EnumCastOpLowering, BytesCastOpLowering,
           ArithBinOpLowering<sol::AddOp, arith::AddIOp>,
           ArithBinOpLowering<sol::SubOp, arith::SubIOp>,
           ArithBinOpLowering<sol::MulOp, arith::MulIOp>,
           ArithBinOpLowering<sol::AndOp, arith::AndIOp>,
           ArithBinOpLowering<sol::OrOp, arith::OrIOp>,
           ArithBinOpLowering<sol::XorOp, arith::XOrIOp>,
           DivOrModOpLowering<sol::DivOp, arith::DivSIOp, arith::DivUIOp>,
           DivOrModOpLowering<sol::ModOp, arith::RemSIOp, arith::RemUIOp>,
           ExpOpLowering, ShrOpLowering, ShlOpLowering, CmpOpLowering,
           AddModOpLowering, MulModOpLowering>(tyConv, pats.getContext());
}

void evm::populateCheckedArithPats(RewritePatternSet &pats,
                                   TypeConverter &tyConv) {
  pats.add<CAddOpLowering, CSubOpLowering, CMulOpLowering, CDivOpLowering,
           CExpOpLowering>(tyConv, pats.getContext());
}

void evm::populateCryptoPats(mlir::RewritePatternSet &pats,
                             mlir::TypeConverter &tyConv) {
  pats.add<Keccak256OpLowering, Sha256OpLowering, Ripemd160OpLowering,
           EcrecoverOpLowering>(tyConv, pats.getContext());
}

void evm::populateMemPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<AllocaOpLowering, MallocOpLowering, ArrayLitOpLowering,
           GetCallDataOpLowering, PushOpLowering, PopOpLowering, GepOpLowering,
           MapOpLowering, LoadOpLowering, LoadImmutableOpLowering,
           StoreOpLowering, DataLocCastOpLowering, LengthOpLowering,
           SliceOpLowering, CopyOpLowering, PushStringOpLowering,
           StringLitOpLowering, ConcatOpLowering>(tyConv, pats.getContext());
  pats.add<AddrOfOpLowering>(pats.getContext());
}

void evm::populateControlFlowPats(RewritePatternSet &pats) {
  pats.add<IfOpLowering, SwitchOpLowering, LoopOpInterfaceLowering>(
      pats.getContext());
}

void evm::populateFuncPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<CallOpLowering, ReturnOpLowering, FuncOpLowering>(tyConv,
                                                             pats.getContext());
}

void evm::populateAddrPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ThisOpLowering, OriginOpLowering, GasPriceOpLowering,
           CallValueOpLowering, BaseFeeOpLowering, BlobBaseFeeOpLowering,
           ChainIdOpLowering, CoinbaseOpLowering, DifficultyOpLowering,
           GasLimitOpLowering, BlockNumberOpLowering, PrevRandaoOpLowering,
           TimestampOpLowering, GasLeftOpLowering, LibAddrOpLowering,
           CodeHashOpLowering, CallerOpLowering, BalanceOpLowering>(
      pats.getContext());
  pats.add<SigOpLowering>(tyConv, pats.getContext());
}

void evm::populateAbiPats(mlir::RewritePatternSet &pats,
                          mlir::TypeConverter &tyConv) {
  pats.add<EncodeOpLowering, DecodeOpLowering>(tyConv, pats.getContext());
}

void evm::populateExtCallPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ExtCallOpLowering, BareCallOpLowering, BareDelegateCallOpLowering,
           BareStaticCallOpLowering, TryOpLowering, NewOpLowering,
           CodeOpLowering, ObjectCodeOpLowering, SendOpLowering,
           TransferOpLowering>(tyConv, pats.getContext());
}

void evm::populateEmitPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<EmitOpLowering, RevertOpLowering>(tyConv, pats.getContext());
}

void evm::populateRequirePat(RewritePatternSet &pats) {
  pats.add<RequireOpLowering, AssertOpLowering>(pats.getContext());
}

void evm::populateContractPat(RewritePatternSet &pats) {
  pats.add<ContractOpLowering>(pats.getContext());
}

void evm::populateStage1Pats(RewritePatternSet &pats, TypeConverter &tyConv) {
  populateArithPats(pats, tyConv);
  populateCheckedArithPats(pats, tyConv);
  populateCryptoPats(pats, tyConv);
  populateMemPats(pats, tyConv);
  populateAddrPat(pats, tyConv);
  populateAbiPats(pats, tyConv);
  populateExtCallPat(pats, tyConv);
  populateEmitPat(pats, tyConv);
  populateRequirePat(pats);
  populateControlFlowPats(pats);
}

void evm::populateStage2Pats(RewritePatternSet &pats) {
  populateContractPat(pats);
  populateYulPats(pats);
}

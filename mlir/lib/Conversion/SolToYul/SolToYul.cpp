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

#include "mlir/Conversion/SolToYul/SolToYul.h"
#include "mlir/Conversion/SolToYul/EVMConstants.h"
#include "mlir/Conversion/SolToYul/EVMUtil.h"
#include "mlir/Conversion/SolToYul/Util.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>
#include <utility>

using namespace mlir;

namespace {

// Compute the types to hand to the generic ABI helpers.
//
// For normal ABI calls we keep the original source types, because the encoder
// still needs to know where values come from (for example storage or calldata
// when copying into memory ABI form).
//
// Library ABI is special only when the boundary type itself is a storage ref,
// in which case we need to use the uint256 type.
SmallVector<Type> getABITypes(OpBuilder &b, TypeRange sourceTys,
                              TypeRange boundaryTys, bool useLibraryABI) {
  assert(sourceTys.size() == boundaryTys.size() &&
         "expected source and boundary types to align");

  if (!useLibraryABI)
    return SmallVector<Type>(sourceTys);

  SmallVector<Type> mappedTys;
  mappedTys.reserve(boundaryTys.size());
  for (auto [sourceTy, boundaryTy] : llvm::zip(sourceTys, boundaryTys)) {
    if (sol::getDataLocation(boundaryTy) == sol::DataLocation::Storage)
      mappedTys.push_back(b.getIntegerType(256));
    else
      mappedTys.push_back(sourceTy);
  }
  return mappedTys;
}

template <typename OpT>
static ModuleOp getModule(OpT op) {
  ModuleOp mod = op->template getParentOfType<ModuleOp>();
  assert(mod && "expected attached module");
  return mod;
}

template <typename OpT, typename Derived>
struct CleanedOperandsLowering : public OpConversionPattern<OpT> {
  using Base = OpConversionPattern<OpT>;
  using OpAdaptor = typename Base::OpAdaptor;
  using Base::Base;

  LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    SmallVector<Value> cleaned;
    cleaned.reserve(adaptor.getOperands().size());
    for (auto [original, converted] :
         llvm::zip(op->getOperands(), adaptor.getOperands()))
      cleaned.push_back(evmB.genCleanup(original.getType(), converted, loc));

    return dispatch(op, r, cleaned);
  }

private:
  template <typename D = Derived>
  LogicalResult dispatch(OpT op, ConversionPatternRewriter &r,
                         ArrayRef<Value> cleaned) const {
    using FnT = decltype(&D::rewriteCleaned);
    if constexpr (std::is_invocable_r_v<LogicalResult, FnT, const D *, OpT,
                                        ConversionPatternRewriter &,
                                        ArrayRef<Value>>) {
      return static_cast<D const *>(this)->rewriteCleaned(op, r, cleaned);
    } else {
      using Traits = llvm::function_traits<FnT>;

      // rewriteCleaned(op, r, cleaned...)
      constexpr size_t fixedArgs = 2;
      static_assert(Traits::num_args >= fixedArgs,
                    "rewriteCleaned is missing required prefix arguments");
      constexpr size_t numCleaned = Traits::num_args - fixedArgs;

      assert(cleaned.size() == numCleaned &&
             "rewriteCleaned signature must match op operand count");

      return dispatchImpl<D>(op, r, cleaned,
                             std::make_index_sequence<numCleaned>{});
    }
  }

  template <typename D, size_t... I>
  LogicalResult dispatchImpl(OpT op, ConversionPatternRewriter &r,
                             ArrayRef<Value> cleaned,
                             std::index_sequence<I...>) const {
    return static_cast<D const *>(this)->rewriteCleaned(op, r, cleaned[I]...);
  }
};

struct ConstantOpLowering : public OpRewritePattern<sol::ConstantOp> {
  using OpRewritePattern<sol::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sol::ConstantOp op,
                                PatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    APInt value = cast<IntegerAttr>(op.getValue()).getValue();
    auto srcTy = cast<IntegerType>(op.getType());

    assert(value.getBitWidth() <= 256 && "Expected constant width <= 256");
    if (value.getBitWidth() < 256)
      value = srcTy.isSigned() ? value.sext(256) : value.zext(256);

    r.replaceOp(op, bExt.genI256Const(value));
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

struct ExtFuncConstantOpLowering
    : public OpConversionPattern<sol::ExtFuncConstantOp> {
  using OpConversionPattern<sol::ExtFuncConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtFuncConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    // Combine addr + selector into MSB-aligned i256:
    // ((addr << 32) | selector) << 64
    Value addr = adaptor.getAddr();
    Value addrShifted =
        r.create<yul::ArithShlOp>(loc, addr, bExt.genI256Const(32));
    Value selector = bExt.genI256Const(op.getSelector());
    Value combined = r.create<yul::OrOp>(loc, addrShifted, selector);
    Value result =
        r.create<yul::ArithShlOp>(loc, combined, bExt.genI256Const(64));
    r.replaceOp(op, result);
    return success();
  }
};

struct ExtFuncSelectorOpLowering
    : public OpConversionPattern<sol::ExtFuncSelectorOp> {
  using OpConversionPattern<sol::ExtFuncSelectorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtFuncSelectorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    // ExtFuncRef is packed as:
    //   | addr (160) | selector (32) | zeros (64) |
    // and bytes4 is represented as the selector in the top 32 bits.
    // Shift left to move the selector into the MSB-aligned bytes4 position.
    r.replaceOpWithNewOp<yul::ArithShlOp>(op, adaptor.getFunc(),
                                          bExt.genI256Const(160));
    return success();
  }
};

struct ExtFuncAddrOpLowering : public OpConversionPattern<sol::ExtFuncAddrOp> {
  using OpConversionPattern<sol::ExtFuncAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtFuncAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    // ExtFuncRef is packed as:
    //   | addr (160) | selector (32) | zeros (64) |
    // The address occupies bits [255:96]; shift right by 96 to right-align it
    // into a clean 160-bit address (high bits zero-filled).
    r.replaceOpWithNewOp<yul::ArithShrOp>(op, adaptor.getFunc(),
                                          bExt.genI256Const(96));
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<sol::CastOp> {
  using OpConversionPattern<sol::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);
    auto inpTyWidth = cast<IntegerType>(op.getInp().getType()).getWidth();
    auto outTyWidth = cast<IntegerType>(op.getType()).getWidth();

    Value val = adaptor.getInp();
    // old codegen begin
    // Match old codegen's plain conversion behavior: widening has to clean the
    // source width first, while narrowing is just a typed view. Semantic
    // consumers such as comparisons, ABI, returns, and stores perform cleanup
    // at their own boundaries.
    // (Via-IR cleans in both directions).
    if (outTyWidth > inpTyWidth)
      val = evmB.genCleanup(op.getInp().getType(), val, loc);
    // old codegen end

    r.replaceOp(op, val);
    return success();
  }
};

struct AddressCastOpLowering : public OpConversionPattern<sol::AddressCastOp> {
  using OpConversionPattern<sol::AddressCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::AddressCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    Type inpTy = op.getInp().getType();
    Type outTy = op.getType();

    if (isa<sol::FixedBytesType>(inpTy) || isa<sol::FixedBytesType>(outTy)) {
      // bytes20 -> address
      if (auto inpBytesTy = dyn_cast<sol::FixedBytesType>(inpTy)) {
        assert(inpBytesTy.getSize() == 20 &&
               "AddressCastOp only supports bytes20");
        assert(isa<sol::AddressType>(outTy));
        r.replaceOpWithNewOp<yul::ArithShrOp>(op, adaptor.getInp(),
                                              bExt.genI256Const(96));
        return success();
      }

      // address -> bytes20
      assert(cast<sol::FixedBytesType>(outTy).getSize() == 20 &&
             "AddressCastOp only supports bytes20");
      assert(isa<sol::AddressType>(inpTy));
      r.replaceOpWithNewOp<yul::ArithShlOp>(op, adaptor.getInp(),
                                            bExt.genI256Const(96));
      return success();
    }

    // old codegen begin
    // (Via-IR adds cleanup helpers around address casts),
    if (isa<IntegerType>(inpTy) || isa<IntegerType>(outTy)) {
      // uint160 -> address
      if (isa<IntegerType>(inpTy)) {
        [[maybe_unused]] auto inpIntTy = cast<IntegerType>(inpTy);
        assert(inpIntTy.getWidth() == 160 && !inpIntTy.isSigned() &&
               "AddressCastOp only supports uint160 -> address");
        assert(isa<sol::AddressType>(outTy));
      } else {
        // address -> uint160
        [[maybe_unused]] auto outIntTy = cast<IntegerType>(outTy);
        assert(isa<sol::AddressType>(inpTy) && outIntTy.getWidth() == 160 &&
               !outIntTy.isSigned() &&
               "AddressCastOp only supports address -> uint160");
      }
      r.replaceOp(op, adaptor.getInp());
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

    r.replaceOp(op, adaptor.getInp());
    // old codegen end
    return success();
  }
};

struct ContractCastOpLowering
    : public OpConversionPattern<sol::ContractCastOp> {
  using OpConversionPattern<sol::ContractCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ContractCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    // old codegen begin
    // (Via-IR adds cleanup helpers around contract casts).
    r.replaceOp(op, adaptor.getInp());
    // old codegen end
    return success();
  }
};

// When doing int -> enum cast, we need to do a cleanup on input and output
// type. For enum -> int, there is no need to cleanup on the output type, as the
// enum value is always in the valid range and thus also valid as an int value.
struct EnumCastOpLowering
    : public CleanedOperandsLowering<sol::EnumCastOp, EnumCastOpLowering> {
  using Base = CleanedOperandsLowering<sol::EnumCastOp, EnumCastOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::EnumCastOp op, ConversionPatternRewriter &r,
                               Value val) const {
    if (isa<sol::EnumType>(op.getType())) {
      Location loc = op.getLoc();
      evm::Builder evmB(getModule(op), r, loc);
      val = evmB.genCleanup(op.getType(), val, loc);
    }
    r.replaceOp(op, val);
    return success();
  }
};

struct BytesCastOpLowering : public OpConversionPattern<sol::BytesCastOp> {
  using OpConversionPattern<sol::BytesCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BytesCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);
    Type inpTy = op.getInp().getType();
    Type outTy = op.getType();

    const bool isByteToBytes1 = isa<sol::ByteType>(inpTy) &&
                                isa<sol::FixedBytesType>(outTy) &&
                                sol::getNumBytes(outTy) == 1;
    const bool isBytes1ToByte = isa<sol::FixedBytesType>(inpTy) &&
                                sol::getNumBytes(inpTy) == 1 &&
                                isa<sol::ByteType>(outTy);
    if (isByteToBytes1 || isBytes1ToByte) {
      // byte <-> bytes1 is a no-op.
      r.replaceOp(op, adaptor.getInp());
      return success();
    }

    // Bytes to bytes
    if (sol::isBytesLikeType(inpTy)) {
      unsigned inpBytesSize = sol::getNumBytes(inpTy);
      if (sol::isBytesLikeType(outTy)) {
        // old codegen begin
        // bytesN -> bytesM shortening is intentionally dirty for old-codegen
        // stack-slot parity. Widening must clear bytes beyond the source width.
        // (Via-IR cleans in both directions).
        Value val = adaptor.getInp();
        if (inpBytesSize < sol::getNumBytes(outTy))
          val = evmB.genCleanup(inpTy, val, loc);
        // old codegen end

        r.replaceOp(op, val);
        return success();
      }

      // Bytes to int
      auto shiftAmt = bExt.genI256Const(256 - (8 * inpBytesSize));
      r.replaceOpWithNewOp<yul::ArithShrOp>(op, adaptor.getInp(), shiftAmt);
      return success();
    }

    // Int to bytes
    assert(isa<IntegerType>(inpTy));
    unsigned outBytesSize = sol::getNumBytes(outTy);
    auto shiftAmt = bExt.genI256Const(256 - (8 * outBytesSize));
    r.replaceOpWithNewOp<yul::ArithShlOp>(op, adaptor.getInp(), shiftAmt);
    return success();
  }
};

struct DynBytesToFixedBytesOpLowering
    : public OpConversionPattern<sol::DynBytesToFixedBytesOp> {
  using OpConversionPattern<sol::DynBytesToFixedBytesOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DynBytesToFixedBytesOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    evm::Builder evmB(getModule(op), r, op.getLoc());
    auto dstTy = cast<sol::FixedBytesType>(op.getType());
    r.replaceOp(op, evmB.genDynBytesToFixedBytes(adaptor.getInp(),
                                                 op.getInp().getType(), dstTy,
                                                 op.getLoc()));
    return success();
  }
};

// old codegen begin
// (Via-IR cleans output of these ops).
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

struct NotOpLowering : public OpConversionPattern<sol::NotOp> {
  using OpConversionPattern<sol::NotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::NotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::NotOp>(op, adaptor.getValue());
    return success();
  }
};
// old codegen end

/// A templatized version of a conversion pattern for lowering div and mod ops.
template <typename SolOp, typename ArithSignedOp, typename ArithUnsignedOp>
struct DivOrModOpLowering
    : public CleanedOperandsLowering<
          SolOp, DivOrModOpLowering<SolOp, ArithSignedOp, ArithUnsignedOp>> {
  using Base = CleanedOperandsLowering<
      SolOp, DivOrModOpLowering<SolOp, ArithSignedOp, ArithUnsignedOp>>;

  using Base::Base;

  LogicalResult rewriteCleaned(SolOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto ty = cast<IntegerType>(op.getType());

    auto zero = bExt.genI256Const(0, loc);
    auto rhsEqZero = bExt.genCmp(yul::CmpPredicate::eq, rhs, zero);
    evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned())
      r.replaceOpWithNewOp<ArithSignedOp>(op, lhs, rhs);
    else
      r.replaceOpWithNewOp<ArithUnsignedOp>(op, lhs, rhs);
    return success();
  }
};

struct ExpOpLowering
    : public CleanedOperandsLowering<sol::ExpOp, ExpOpLowering> {
  using Base = CleanedOperandsLowering<sol::ExpOp, ExpOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::ExpOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    r.replaceOpWithNewOp<yul::ExpOp>(op, lhs, rhs);
    return success();
  }
};

struct CExpOpLowering
    : public CleanedOperandsLowering<sol::CExpOp, CExpOpLowering> {
  using Base = CleanedOperandsLowering<sol::CExpOp, CExpOpLowering>;
  using Base::Base;

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

  std::pair<Value, Value> genExpHelper(ModuleOp mod,
                                       ConversionPatternRewriter &r,
                                       Location loc, Value initPow,
                                       Value initBase, Value initExp,
                                       Value maxPow) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(mod, r, loc);

    Value one = bExt.genI256Const(1, loc);
    SmallVector<Value> whileInitVals{initPow, initBase, initExp};
    TypeRange whileArgTypes(whileInitVals);
    SmallVector<Location> whileBbArgLocs(whileInitVals.size(), loc);

    // Produces power, base
    auto whileOp = r.create<yul::WhileOp>(loc, whileArgTypes, whileInitVals);
    // Condition region
    {
      Block *condBlock = &whileOp.getCond().emplaceBlock();
      whileOp.getCond().addArguments(whileArgTypes, whileBbArgLocs);
      r.setInsertionPointToStart(condBlock);
      Value curPow = condBlock->getArgument(0);
      Value curBase = condBlock->getArgument(1);
      Value curExp = condBlock->getArgument(2);
      Value cmp = bExt.genCmp(yul::CmpPredicate::ugt, curExp, one);
      r.create<yul::ConditionOp>(loc, cmp, ValueRange{curPow, curBase, curExp});
    }

    // Body region
    {
      Block *bodyBlock = &whileOp.getBody().emplaceBlock();
      whileOp.getBody().addArguments(whileArgTypes, whileBbArgLocs);
      r.setInsertionPointToStart(bodyBlock);
      Value curPow = bodyBlock->getArgument(0);
      Value curBase = bodyBlock->getArgument(1);
      Value curExp = bodyBlock->getArgument(2);
      {
        Value tmpDiv = r.create<yul::DivOp>(loc, maxPow, curBase);
        Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, curBase, tmpDiv);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
      }
      Value lsb = r.create<yul::AndOp>(loc, curExp, one);
      Value zero = bExt.genI256Const(0, loc);
      Value expIsOdd = bExt.genCmp(yul::CmpPredicate::ne, lsb, zero);
      Value newPow = r.create<yul::ArithSelectOp>(
                          loc, expIsOdd,
                          r.create<yul::MulOp>(loc, curPow, curBase), curPow)
                         .getResult();
      Value newBase = r.create<yul::MulOp>(loc, curBase, curBase);
      Value newExp = r.create<yul::ArithShrOp>(loc, curExp, one);
      r.create<yul::YieldOp>(loc, ValueRange{newPow, newBase, newExp});
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

  Value expUnsigned(sol::CExpOp op, ConversionPatternRewriter &r, Value base,
                    Value exp) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    IntegerType i256Ty = r.getIntegerType(256);
    auto ty = cast<IntegerType>(op.getType());
    Value max256 = bExt.genI256Const(
        llvm::APInt::getMaxValue(ty.getWidth()).zext(256), loc);
    Value zero = bExt.genI256Const(0, loc);
    Value one = bExt.genI256Const(1, loc);

    auto expEqZero = bExt.genCmp(yul::CmpPredicate::eq, exp, zero);
    auto baseEqOne = bExt.genCmp(yul::CmpPredicate::eq, base, one);
    auto powOneCond = r.create<yul::OrOp>(loc, expEqZero, baseEqOne);

    // If 1 begin
    auto ifExpEqZero = bExt.createIf(i256Ty, powOneCond);
    // If 1 then
    r.setInsertionPointToStart(&ifExpEqZero.getThenRegion().front());
    r.create<yul::YieldOp>(loc, one);

    // If 1 else
    r.setInsertionPointToStart(&ifExpEqZero.getElseRegion().front());
    auto baseEqZero = bExt.genCmp(yul::CmpPredicate::eq, base, zero);

    // If 2 begin
    auto ifBaseEqZero = bExt.createIf(i256Ty, baseEqZero);
    // If 2 then
    r.setInsertionPointToStart(&ifBaseEqZero.getThenRegion().front());
    r.create<yul::YieldOp>(loc, zero);

    // If 2 else
    r.setInsertionPointToStart(&ifBaseEqZero.getElseRegion().front());
    Value two = bExt.genI256Const(2, loc);
    auto baseEqTwo = bExt.genCmp(yul::CmpPredicate::eq, base, two);

    // If 3 begin
    auto ifBaseEqTwo = bExt.createIf(i256Ty, baseEqTwo);
    // If 3 then
    r.setInsertionPointToStart(&ifBaseEqTwo.getThenRegion().front());
    {
      auto expGt255 =
          bExt.genCmp(yul::CmpPredicate::ugt, exp, bExt.genI256Const(255, loc));
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, expGt255);

      auto two256 = bExt.genI256Const(2, loc);
      Value tmpPow = r.create<yul::ExpOp>(loc, two256, exp);
      auto panicCond = bExt.genCmp(yul::CmpPredicate::ugt, tmpPow, max256);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);

      r.create<yul::YieldOp>(loc, tmpPow);
    }
    // If 3 else
    r.setInsertionPointToStart(&ifBaseEqTwo.getElseRegion().front());
    auto baseLT11 =
        bExt.genCmp(yul::CmpPredicate::ult, base, bExt.genI256Const(11, loc));
    Value baseLT307 = (ty.getWidth() > 8)
                          ? bExt.genCmp(yul::CmpPredicate::ult, base,
                                        bExt.genI256Const(307, loc))
                          : bExt.genBool(true, loc);

    auto expLT78 =
        bExt.genCmp(yul::CmpPredicate::ult, exp, bExt.genI256Const(78, loc));
    auto expLT32 =
        bExt.genCmp(yul::CmpPredicate::ult, exp, bExt.genI256Const(32, loc));

    auto viaExpBuitinCond =
        r.create<yul::OrOp>(loc, r.create<yul::AndOp>(loc, baseLT11, expLT78),
                            r.create<yul::AndOp>(loc, baseLT307, expLT32));
    // If 4 begin
    auto ifViaExpBuitinCond = bExt.createIf(i256Ty, viaExpBuitinCond);
    // If 4 then
    r.setInsertionPointToStart(&ifViaExpBuitinCond.getThenRegion().front());

    {
      Value tmpPow = r.create<yul::ExpOp>(loc, base, exp);
      Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, tmpPow, max256);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);

      r.create<yul::YieldOp>(loc, tmpPow);
    }
    // If 4 else
    r.setInsertionPointToStart(&ifViaExpBuitinCond.getElseRegion().front());
    auto [pow2, base2] =
        genExpHelper(getModule(op), r, loc, one, base, exp, max256);

    {
      r.setInsertionPointToEnd(&ifViaExpBuitinCond.getElseRegion().front());
      Value tmpDiv = r.create<yul::DivOp>(loc, max256, base2);
      Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, pow2, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
      Value res = r.create<yul::MulOp>(loc, base2, pow2);
      r.create<yul::YieldOp>(loc, res);
    }
    // If 4 (ifViaExpBuitinCond) end
    r.setInsertionPointToEnd(&ifBaseEqTwo.getElseRegion().front());
    r.create<yul::YieldOp>(loc, ifViaExpBuitinCond.getResult(0));
    // If 3 (ifBaseEqTwo) end

    r.setInsertionPointToEnd(&ifBaseEqZero.getElseRegion().front());
    r.create<yul::YieldOp>(loc, ifBaseEqTwo.getResult(0));
    // If 2 (ifBaseEqZero) end

    r.setInsertionPointToEnd(&ifExpEqZero.getElseRegion().front());
    r.create<yul::YieldOp>(loc, ifBaseEqZero.getResult(0));
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

  Value expSigned(sol::CExpOp op, ConversionPatternRewriter &r, Value base,
                  Value exp) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    IntegerType i256Ty = r.getIntegerType(256);
    auto ty = cast<IntegerType>(op.getType());
    Value zero = bExt.genI256Const(0, loc);
    Value one = bExt.genI256Const(1, loc);
    Value max256 = bExt.genI256Const(
        llvm::APInt::getSignedMaxValue(ty.getWidth()).sext(256), loc);
    Value min256 = bExt.genI256Const(
        llvm::APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);

    auto expEqZero = bExt.genCmp(yul::CmpPredicate::eq, exp, zero);

    // If 1 begin
    auto ifExpEqZero = bExt.createIf(i256Ty, expEqZero);
    // If 1 then
    r.setInsertionPointToStart(&ifExpEqZero.getThenRegion().front());
    r.create<yul::YieldOp>(loc, one);

    // If 1 else
    r.setInsertionPointToStart(&ifExpEqZero.getElseRegion().front());
    auto baseEqZero = bExt.genCmp(yul::CmpPredicate::eq, base, zero);

    // If 2 begin
    auto ifBaseEqZero = bExt.createIf(i256Ty, baseEqZero);
    // If 2 then
    r.setInsertionPointToStart(&ifBaseEqZero.getThenRegion().front());
    r.create<yul::YieldOp>(loc, zero);

    // If 2 else
    r.setInsertionPointToStart(&ifBaseEqZero.getElseRegion().front());
    auto expEqOne = bExt.genCmp(yul::CmpPredicate::eq, exp, one);

    // If 3 begin
    auto ifExpEqOne = bExt.createIf(i256Ty, expEqOne);
    // If 3 then
    r.setInsertionPointToStart(&ifExpEqOne.getThenRegion().front());
    r.create<yul::YieldOp>(loc, base);

    // If 3 else
    r.setInsertionPointToStart(&ifExpEqOne.getElseRegion().front());
    auto baseSgtZero = bExt.genCmp(yul::CmpPredicate::sgt, base, zero);

    // If 4 begin
    auto ifBaseSgtZero = bExt.createIf(TypeRange{}, baseSgtZero,
                                       /*withElseRegion=*/true);
    // If 4 then
    r.setInsertionPointToStart(&ifBaseSgtZero.getThenRegion().front());
    {
      Value tmpDiv = r.create<yul::DivOp>(loc, max256, base);
      Value panicCond = bExt.genCmp(yul::CmpPredicate::ugt, base, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    // If 4 else
    r.setInsertionPointToStart(&ifBaseSgtZero.getElseRegion().front());
    {
      Value tmpDiv = r.create<yul::SDivOp>(loc, max256, base);
      Value panicCond = bExt.genCmp(yul::CmpPredicate::slt, base, tmpDiv);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    // If 4 end
    r.setInsertionPointAfter(ifBaseSgtZero);

    Value lsb = r.create<yul::AndOp>(loc, exp, one);
    Value expIsOdd = bExt.genCmp(yul::CmpPredicate::ne, lsb, zero);
    Value pow2 =
        r.create<yul::ArithSelectOp>(loc, expIsOdd, base, one).getResult();
    Value base2 = r.create<yul::MulOp>(loc, base, base);
    Value exp2 = r.create<yul::ArithShrOp>(loc, exp, one);
    auto [pow3, base3] =
        genExpHelper(getModule(op), r, loc, pow2, base2, exp2, max256);

    r.setInsertionPointToEnd(&ifExpEqOne.getElseRegion().front());
    {
      auto zero256 = bExt.genI256Const(0, loc);
      Value div = r.create<yul::DivOp>(loc, max256, base3);
      Value cmp = bExt.genCmp(yul::CmpPredicate::ugt, pow3, div);
      Value cmp2 = bExt.genCmp(yul::CmpPredicate::sgt, pow3, zero256);
      Value panicCond = r.create<yul::AndOp>(loc, cmp, cmp2);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }
    {
      auto zero256 = bExt.genI256Const(0, loc);
      Value div = r.create<yul::SDivOp>(loc, min256, base3);
      Value cmp = bExt.genCmp(yul::CmpPredicate::slt, pow3, div);
      Value cmp2 = bExt.genCmp(yul::CmpPredicate::slt, pow3, zero256);
      Value panicCond = r.create<yul::AndOp>(loc, cmp, cmp2);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
    }

    Value pow4 = r.create<yul::MulOp>(loc, pow3, base3);
    r.create<yul::YieldOp>(loc, pow4);
    // If 3 (ifExpEqOne) end

    r.setInsertionPointToEnd(&ifBaseEqZero.getElseRegion().front());
    r.create<yul::YieldOp>(loc, ifExpEqOne.getResult(0));
    // If 2 (ifBaseEqZero) end

    r.setInsertionPointToEnd(&ifExpEqZero.getElseRegion().front());
    r.create<yul::YieldOp>(loc, ifBaseEqZero.getResult(0));
    // If 1 (ifExpEqZero) end

    r.setInsertionPointAfter(ifExpEqZero);

    return ifExpEqZero.getResult(0);
  }

  LogicalResult rewriteCleaned(sol::CExpOp op, ConversionPatternRewriter &r,
                               Value base, Value exp) const {
    auto ty = cast<IntegerType>(op.getType());
    Value res = ty.isSigned() ? expSigned(op, r, base, exp)
                              : expUnsigned(op, r, base, exp);
    r.replaceOp(op, res);

    return success();
  }
};

} // namespace

template <typename OpT>
static Value getCryptoHashLowering(OpT op, uint32_t preCompieAddr,
                                   typename OpT::Adaptor adaptor,
                                   ConversionPatternRewriter &r) {
  Location loc = op.getLoc();
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(getModule(op), r, loc);

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

  auto statusIsZero = bExt.genCmp(yul::CmpPredicate::eq, status, zero);
  evmB.genForwardingRevert(statusIsZero);

  auto res = evmB.genLoad(zero, dataLoc);
  // For ripemd160 the result must be shifted left by 96 bits.
  if constexpr (std::is_same_v<std::decay_t<OpT>, sol::Ripemd160Op>)
    res = r.create<yul::ShlOp>(loc, bExt.genI256Const(96), res);

  return res;
}

template <typename ModOpT>
static Value genYulModOp(ModuleOp module, ConversionPatternRewriter &r,
                         Location loc, Value x, Value y, Value mod) {
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(module, r, loc);

  auto zero = bExt.genI256Const(0);
  auto modEqZero = bExt.genCmp(yul::CmpPredicate::eq, mod, zero);
  evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, modEqZero);
  return r.create<ModOpT>(loc, x, y, mod);
}

namespace {

struct ShlOpLowering
    : public CleanedOperandsLowering<sol::ShlOp, ShlOpLowering> {
  using Base = CleanedOperandsLowering<sol::ShlOp, ShlOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::ShlOp op, ConversionPatternRewriter &r,
                               Value val, Value shift) const {
    r.replaceOpWithNewOp<yul::ShlOp>(op, shift, val);
    return success();
  }
};

struct ShrOpLowering
    : public CleanedOperandsLowering<sol::ShrOp, ShrOpLowering> {
  using Base = CleanedOperandsLowering<sol::ShrOp, ShrOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::ShrOp op, ConversionPatternRewriter &r,
                               Value val, Value shift) const {
    auto ty = cast<IntegerType>(op.getType());
    if (ty.isSigned())
      r.replaceOpWithNewOp<yul::SarOp>(op, shift, val);
    else
      r.replaceOpWithNewOp<yul::ShrOp>(op, shift, val);
    return success();
  }
};

struct CAddOpLowering
    : public CleanedOperandsLowering<sol::CAddOp, CAddOpLowering> {
  using Base = CleanedOperandsLowering<sol::CAddOp, CAddOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CAddOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value sum = r.create<yul::AddOp>(loc, lhs, rhs);

    if (ty.getWidth() < 256) {
      if (ty.isSigned()) {
        // overflow, if sum > max
        // underflow, if sum < min
        Value max = bExt.genI256Const(
            APInt::getSignedMaxValue(ty.getWidth()).sext(256), loc);
        Value overMax = bExt.genCmp(yul::CmpPredicate::sgt, sum, max);
        Value min = bExt.genI256Const(
            APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);
        Value underMin = bExt.genCmp(yul::CmpPredicate::slt, sum, min);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                      r.create<yul::OrOp>(loc, overMax, underMin));
      } else {
        // overflow, if sum > max
        Value max =
            bExt.genI256Const(APInt::getMaxValue(ty.getWidth()).zext(256), loc);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                      bExt.genCmp(yul::CmpPredicate::ugt, sum, max));
      }
      r.replaceOp(op, sum);
      return success();
    }

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // overflow, if y >= 0 and sum < x
      // underflow, if y < 0 and sum >= x
      //
      // We compare rhs with zero since the canonicalizer could make the rhs a
      // constant which would enable the arith dialect to optimize away the
      // comparison.

      auto zero = bExt.genI256Const(0, loc);

      // Generate the overflow condition.
      auto rhsGtEqZero = bExt.genCmp(yul::CmpPredicate::sge, rhs, zero);
      auto sumLtLhs = bExt.genCmp(yul::CmpPredicate::slt, sum, lhs);
      auto overflowCond = r.create<yul::AndOp>(loc, rhsGtEqZero, sumLtLhs);

      // Generate the underflow condition.
      auto rhsLtZero = bExt.genCmp(yul::CmpPredicate::slt, rhs, zero);
      auto sumGtEqLhs = bExt.genCmp(yul::CmpPredicate::sge, sum, lhs);
      auto underflowCond = r.create<yul::AndOp>(loc, rhsLtZero, sumGtEqLhs);

      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<yul::OrOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    bExt.genCmp(yul::CmpPredicate::ugt, lhs, sum));
    }

    r.replaceOp(op, sum);
    return success();
  }
};

struct CSubOpLowering
    : public CleanedOperandsLowering<sol::CSubOp, CSubOpLowering> {
  using Base = CleanedOperandsLowering<sol::CSubOp, CSubOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CSubOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto ty = cast<IntegerType>(op.getType());
    Value diff = r.create<yul::SubOp>(loc, lhs, rhs);

    if (ty.getWidth() < 256) {
      if (ty.isSigned()) {
        // overflow, if diff > max
        // underflow, if diff < min
        Value max = bExt.genI256Const(
            APInt::getSignedMaxValue(ty.getWidth()).sext(256), loc);
        Value overMax = bExt.genCmp(yul::CmpPredicate::sgt, diff, max);
        Value min = bExt.genI256Const(
            APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);
        Value underMin = bExt.genCmp(yul::CmpPredicate::slt, diff, min);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                      r.create<yul::OrOp>(loc, overMax, underMin));
      } else {
        // underflow, if diff > max
        Value max =
            bExt.genI256Const(APInt::getMaxValue(ty.getWidth()).zext(256), loc);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                      bExt.genCmp(yul::CmpPredicate::ugt, diff, max));
      }
      r.replaceOp(op, diff);
      return success();
    }

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if y >= 0 and diff > x
      // overflow, if y < 0 and diff < x

      auto zero = bExt.genI256Const(0, loc);

      // Generate the overflow condition.
      auto rhsGtEqZero = bExt.genCmp(yul::CmpPredicate::sge, rhs, zero);
      auto diffGtLhs = bExt.genCmp(yul::CmpPredicate::sgt, diff, lhs);
      auto overflowCond = r.create<yul::AndOp>(loc, rhsGtEqZero, diffGtLhs);

      // Generate the underflow condition.
      auto rhsLtZero = bExt.genCmp(yul::CmpPredicate::slt, rhs, zero);
      auto diffLtRhs = bExt.genCmp(yul::CmpPredicate::slt, diff, lhs);
      auto underflowCond = r.create<yul::AndOp>(loc, rhsLtZero, diffLtRhs);

      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<yul::OrOp>(loc, overflowCond, underflowCond));

      // Unsigned case
    } else {
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    bExt.genCmp(yul::CmpPredicate::ugt, diff, lhs));
    }

    r.replaceOp(op, diff);
    return success();
  }
};

struct CMulOpLowering
    : public CleanedOperandsLowering<sol::CMulOp, CMulOpLowering> {
  using Base = CleanedOperandsLowering<sol::CMulOp, CMulOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CMulOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto ty = cast<IntegerType>(op.getType());

    Value rawProduct = r.create<yul::MulOp>(loc, lhs, rhs);
    Value product = evmB.genCleanup(op.getType(), rawProduct, loc);
    auto zero = bExt.genI256Const(0, loc);
    if (ty.getWidth() <= 128) {
      // overflow/underflow, if product != product_raw
      Value panicCond = bExt.genCmp(yul::CmpPredicate::ne, rawProduct, product);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow, panicCond);
      r.replaceOp(op, product);
      return success();
    }

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x < 0 and y == int.min
      if (ty.getWidth() == 256) {
        auto lhsLtZero = bExt.genCmp(yul::CmpPredicate::slt, lhs, zero);
        auto minVal = bExt.genI256Const(
            APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);
        auto rhsEqMin = bExt.genCmp(yul::CmpPredicate::eq, rhs, minVal);
        evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                      r.create<yul::AndOp>(loc, lhsLtZero, rhsEqMin));
      }

      // over/underflow, if x != 0 and product/x != y
      auto lhsNeqZero = bExt.genCmp(yul::CmpPredicate::ne, lhs, zero);
      auto quotient = r.create<yul::SDivOp>(loc, product, lhs);
      auto quotientNeqRhs = bExt.genCmp(yul::CmpPredicate::ne, quotient, rhs);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<yul::AndOp>(loc, lhsNeqZero, quotientNeqRhs));

      // Unsigned case
    } else {
      // over/underflow, if x != 0 and product/x != y
      auto lhsNeqZero = bExt.genCmp(yul::CmpPredicate::ne, lhs, zero);
      auto quotient = r.create<yul::DivOp>(loc, product, lhs);
      auto quotientNeqRhs = bExt.genCmp(yul::CmpPredicate::ne, quotient, rhs);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<yul::AndOp>(loc, lhsNeqZero, quotientNeqRhs));
    }

    r.replaceOp(op, product);
    return success();
  }
};

struct AddModOpLowering
    : public CleanedOperandsLowering<sol::AddModOp, AddModOpLowering> {
  using Base = CleanedOperandsLowering<sol::AddModOp, AddModOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::AddModOp op, ConversionPatternRewriter &r,
                               Value x, Value y, Value mod) const {
    Location loc = op.getLoc();
    Value result = genYulModOp<yul::AddModOp>(getModule(op), r, loc, x, y, mod);
    r.replaceOp(op, result);
    return success();
  }
};

struct MulModOpLowering
    : public CleanedOperandsLowering<sol::MulModOp, MulModOpLowering> {
  using Base = CleanedOperandsLowering<sol::MulModOp, MulModOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::MulModOp op, ConversionPatternRewriter &r,
                               Value x, Value y, Value mod) const {
    Location loc = op.getLoc();
    Value result = genYulModOp<yul::MulModOp>(getModule(op), r, loc, x, y, mod);
    r.replaceOp(op, result);
    return success();
  }
};

struct CDivOpLowering
    : public CleanedOperandsLowering<sol::CDivOp, CDivOpLowering> {
  using Base = CleanedOperandsLowering<sol::CDivOp, CDivOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CDivOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto ty = cast<IntegerType>(op.getType());
    auto zero = bExt.genI256Const(0, loc);
    auto rhsEqZero = bExt.genCmp(yul::CmpPredicate::eq, rhs, zero);
    evmB.genPanic(mlir::evm::PanicCode::DivisionByZero, rhsEqZero);

    if (ty.isSigned()) {
      // (Copied from the yul codegen)
      // underflow, if x == int.min and y == -1
      auto minVal = bExt.genI256Const(
          APInt::getSignedMinValue(ty.getWidth()).sext(256), loc);
      auto lhsEqMin = bExt.genCmp(yul::CmpPredicate::eq, lhs, minVal);
      auto minusOne = bExt.genI256Const(-1, loc);
      auto rhsEqMinusOne = bExt.genCmp(yul::CmpPredicate::eq, rhs, minusOne);
      evmB.genPanic(mlir::evm::PanicCode::UnderOverflow,
                    r.create<yul::AndOp>(loc, lhsEqMin, rhsEqMinusOne));
      r.replaceOpWithNewOp<yul::ArithSDivOp>(op, lhs, rhs);
    } else {
      r.replaceOpWithNewOp<yul::ArithDivOp>(op, lhs, rhs);
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
    evm::Builder evmB(getModule(op), r, loc);

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
    evm::Builder evmB(getModule(op), r, loc);

    Value zero = bExt.genI256Const(0);
    Value retSize = bExt.genI256Const(32);
    Value freePtr = evmB.genFreePtr();
    // Since the size of the encoded arguments is known in advance,
    // we ignore the return value of genABIEncoding.
    Value paramsSize = bExt.genI256Const(128);
    evmB.genABIEncoding(op.getOperandTypes(), adaptor.getOperands(), freePtr);
    evmB.genFreePtrUpd(freePtr, paramsSize);

    // Hashing functions store their result in scratch space (0x00–0x3f).
    Value gas = r.create<mlir::yul::GasOp>(loc);
    // TODO: It's not clear why do we need this.
    r.create<yul::MStoreOp>(loc, zero, zero);
    mlir::Value status =
        r.create<yul::StaticCallOp>(loc, gas,
                                    /*address=*/bExt.genI256Const(1),
                                    /*inpOffset=*/freePtr, paramsSize,
                                    /*outOffset=*/zero, /*outSize=*/retSize);

    auto statusIsZero = bExt.genCmp(yul::CmpPredicate::eq, status, zero);
    evmB.genForwardingRevert(statusIsZero);
    auto res = evmB.genLoad(zero, sol::DataLocation::Memory);
    r.replaceOp(op, res);

    return success();
  }
};

struct CmpOpLowering
    : public CleanedOperandsLowering<sol::CmpOp, CmpOpLowering> {
  using Base = CleanedOperandsLowering<sol::CmpOp, CmpOpLowering>;
  using Base::Base;

  yul::CmpPredicate getSignlessPred(sol::CmpPredicate pred,
                                    bool isSigned) const {
    // Sign insensitive predicates.
    switch (pred) {
    case sol::CmpPredicate::eq:
      return yul::CmpPredicate::eq;
    case sol::CmpPredicate::ne:
      return yul::CmpPredicate::ne;
    default:
      break;
    }

    // Sign sensitive predicates.
    if (isSigned) {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return yul::CmpPredicate::slt;
      case sol::CmpPredicate::le:
        return yul::CmpPredicate::sle;
      case sol::CmpPredicate::gt:
        return yul::CmpPredicate::sgt;
      case sol::CmpPredicate::ge:
        return yul::CmpPredicate::sge;
      default:
        break;
      }
    } else {
      switch (pred) {
      case sol::CmpPredicate::lt:
        return yul::CmpPredicate::ult;
      case sol::CmpPredicate::le:
        return yul::CmpPredicate::ule;
      case sol::CmpPredicate::gt:
        return yul::CmpPredicate::ugt;
      case sol::CmpPredicate::ge:
        return yul::CmpPredicate::uge;
      default:
        break;
      }
    }
    llvm_unreachable("Invalid predicate");
  }

  LogicalResult rewriteCleaned(sol::CmpOp op, ConversionPatternRewriter &r,
                               Value lhs, Value rhs) const {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    bool isSigned = false;
    if (auto intTy = dyn_cast<IntegerType>(op.getLhs().getType()))
      isSigned = intTy.isSigned();

    yul::CmpPredicate signlessPred =
        getSignlessPred(op.getPredicate(), isSigned);
    r.replaceOp(op, bExt.genCmp(signlessPred, lhs, rhs));
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
    evm::Builder evmB(getModule(op), r, op.getLoc());
    Type sizeTy = op.getSize() ? op.getSize().getType() : Type{};
    r.replaceOp(op, evmB.genMemAlloc(op.getType(), op.getZeroInit(), {},
                                     adaptor.getSize(), sizeTy));
    return success();
  }
};

struct ArrayLitOpLowering : public OpConversionPattern<sol::ArrayLitOp> {
  using OpConversionPattern<sol::ArrayLitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ArrayLitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);
    r.replaceOp(op, evmB.genMemAlloc(op.getType(), false, adaptor.getIns(), {},
                                     Type{}));
    return success();
  }
};

struct StringLitOpLowering : public OpConversionPattern<sol::StringLitOp> {
  using OpConversionPattern<sol::StringLitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StringLitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {

    Location loc = op.getLoc();
    auto mod = getModule(op);
    evm::Builder evmB(mod, r, loc);
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
      allocPtr =
          evmB.genMemAlloc(op.getType(), false, {}, litSize, Type{}, loc);
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
    evm::Builder evmB(getModule(op), r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Value freePtr = evmB.genFreePtr();
    Value dstStart = r.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(32));

    Value currDst = dstStart;
    for (auto [origSrc, src] : llvm::zip(op.getArgs(), adaptor.getArgs())) {
      Type ty = origSrc.getType();
      if (auto bytesTy = dyn_cast<sol::FixedBytesType>(ty)) {
        // Cleanup the src before copying to memory to ensure that any
        // padding bytes are zeroed out.
        r.create<yul::MStoreOp>(loc, currDst, evmB.genCleanup(ty, src, loc));
        Value length = bExt.genI256Const(bytesTy.getSize());
        currDst = r.create<yul::AddOp>(loc, currDst, length);
        continue;
      }

      assert(isa<sol::StringType>(ty));
      Value length = evmB.genCopyStringDataToMemory(src, ty, currDst, loc);
      currDst = r.create<yul::AddOp>(loc, currDst, length);
    }
    Value dataSize = r.create<yul::SubOp>(loc, currDst, dstStart);
    r.create<yul::MStoreOp>(loc, freePtr, dataSize);
    Value allocationSize = r.create<yul::SubOp>(loc, currDst, freePtr);
    evmB.genFreePtrUpd(freePtr, allocationSize);
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

struct GetReturnDataOpLowering
    : public OpConversionPattern<sol::GetReturnDataOp> {
  using OpConversionPattern<sol::GetReturnDataOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::GetReturnDataOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    auto mod = getModule(op);
    evm::Builder evmB(mod, r, loc);

    // Materialise the returndata (from byte offset `start`) as a fresh memory
    // `bytes`, mirroring the (status, returndata) tail of the bare-call
    // lowering: allocate a dynamic array sized to `returndatasize - start` and
    // `returndatacopy` into it from `start`.
    Value start = adaptor.getStart();
    Value retDataSize = r.create<yul::ReturnDataSizeOp>(loc);
    Value copySize = r.create<yul::SubOp>(loc, retDataSize, start);
    Value roundedCopySize = bExt.genRoundUpToMultiple<32>(copySize);
    Value retData =
        evmB.genMemAllocForDynArray(copySize, roundedCopySize, loc);
    Value retDataStart =
        evmB.genDataAddrPtr(retData, sol::DataLocation::Memory, loc);
    r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/retDataStart,
                                    /*src=*/start, copySize);
    r.replaceOp(op, retData);
    return success();
  }
};

struct DefaultCallDataOpLowering
    : public OpConversionPattern<sol::DefaultCallDataOp> {
  using OpConversionPattern<sol::DefaultCallDataOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DefaultCallDataOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    mlir::solgen::BuilderExt bExt(r, op.getLoc());
    Value zero = bExt.genI256Const(0);

    Value callDataSz = r.create<yul::CallDataSizeOp>(op.getLoc());

    // Fat pointer types get an empty slice: {calldatasize(), 0}.
    Type convertedTy = getTypeConverter()->convertType(op.getType());
    if (isa<LLVM::LLVMStructType>(convertedTy)) {
      r.replaceOp(op, bExt.genLLVMStruct({callDataSz, zero}));
      return success();
    }
    // Scalar calldata types (e.g. structs) default to calldatasize() -
    // an offset past valid calldata.
    r.replaceOp(op, callDataSz);
    return success();
  }
};

struct PushOpLowering : public OpConversionPattern<sol::PushOp> {
  using OpConversionPattern<sol::PushOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::PushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Type inpTy = op.getInp().getType();
    if (isa<sol::StringType>(inpTy)) {
      r.replaceOp(op, evmB.genPushVoidToString(adaptor.getInp(), loc));
      return success();
    }

    Value slot = adaptor.getInp();
    Value oldSize = r.create<yul::SLoadOp>(loc, slot);
    Value newSize = r.create<yul::AddOp>(loc, oldSize, bExt.genI256Const(1));
    r.create<yul::SStoreOp>(loc, slot, newSize);
    Value dataSlot = evmB.genDataAddrPtr(slot, sol::DataLocation::Storage);

    // Get element type from the input type.
    Type eltTy;
    if (auto arrTy = dyn_cast<sol::ArrayType>(inpTy)) {
      eltTy = arrTy.getEltType();
    } else {
      llvm_unreachable("");
    }

    if (sol::canBePacked(eltTy)) {
      r.replaceOp(op, evmB.genPackedStorageAddr(dataSlot, oldSize, eltTy));
    } else {
      // Slot-aligned layout.
      Value stride = bExt.genI256Const(sol::getStorageSlotCount(eltTy));
      Value scaledIdx = r.create<yul::MulOp>(loc, oldSize, stride);
      r.replaceOp(op, r.create<yul::AddOp>(loc, dataSlot, scaledIdx));
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
    evm::Builder evmB(getModule(op), r, loc);

    Type inpValueTy = op.getValue().getType();
    Value val = adaptor.getValue();

    Value byte =
        isa<sol::FixedBytesType>(inpValueTy)
            ? r.create<yul::ArithShrOp>(loc, val, bExt.genI256Const(248))
                  .getResult()
            : val;
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
    evm::Builder evmB(getModule(op), r, loc);
    mlir::solgen::BuilderExt bExt(r, loc);

    Type inpTy = op.getInp().getType();
    Value slot = adaptor.getInp();
    Value data = evmB.genLoad(slot, sol::DataLocation::Storage, loc);
    Value oldSize = isa<sol::StringType>(inpTy)
                        ? evmB.genStorageStringLength(data, loc)
                        : data;

    // Generate the empty array panic check.
    Value panicCond =
        bExt.genCmp(yul::CmpPredicate::eq, oldSize, bExt.genI256Const(0));
    evmB.genPanic(mlir::evm::PanicCode::EmptyArrayPop, panicCond);

    if (isa<sol::StringType>(inpTy)) {
      evmB.genPopString(slot, data, oldSize, loc);
      r.eraseOp(op);
      return success();
    }
    Value newSize = r.create<yul::SubOp>(loc, oldSize, bExt.genI256Const(1));
    auto arrTy = cast<sol::ArrayType>(inpTy);
    // Yul reference order. genClearStorageArrayTail handles both packed types
    // and slot-aligned types (including multi-slot structs and nested
    // dynamic arrays).
    // isDecrement=true: newSize = oldSize-1, so the range is always non-empty
    // and contains exactly one element; the range guard and loop are omitted.
    evmB.genClearStorageArrayTail(slot, arrTy, newSize, oldSize,
                                  /*isDecrement=*/true, loc);

    // Write the decremented length after clearing the removed element.
    r.create<yul::SStoreOp>(loc, slot, newSize);

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
      if (sol::canBePacked(stateVarOp.getType())) {
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
    evm::Builder evmB(getModule(op), r, loc);

    Type baseAddrTy = op.getBaseAddr().getType();
    Value remappedBaseAddr = adaptor.getBaseAddr();
    Value idx = adaptor.getIdx();
    Value cleanedIdx;
    auto getCleanedIdx = [&]() -> Value {
      if (!cleanedIdx)
        cleanedIdx = evmB.genCleanup(op.getIdx().getType(), idx, loc);
      return cleanedIdx;
    };
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
        assert((isa<IntegerType>(eltTy) || isa<sol::FixedBytesType>(eltTy) ||
                sol::isAddressLikeType(eltTy) || isa<sol::EnumType>(eltTy) ||
                isa<sol::FuncRefType>(eltTy) ||
                isa<sol::ExtFuncRefType>(eltTy) ||
                sol::isNonPtrRefType(eltTy)) &&
               "NYI");

        // Bounds check (always for dynamic, non-const index for static).
        bool isConstIdx = !isa<BlockArgument>(idx) &&
                          isa<yul::ConstantOp>(idx.getDefiningOp());
        Value arrayIdx = isConstIdx ? idx : getCleanedIdx();
        if (arrTy.isDynSized() || !isConstIdx) {
          Value size = arrTy.isDynSized()
                           ? evmB.genDynSize(remappedBaseAddr, baseAddrTy)
                           : bExt.genI256Const(arrTy.getSize());
          auto panicCond = bExt.genCmp(yul::CmpPredicate::uge, arrayIdx, size);
          evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);
        }

        // Base slot: keccak256(slot) for dynamic, slot for static.
        Value baseSlot = arrTy.isDynSized()
                             ? evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy)
                             : remappedBaseAddr;

        if (sol::canBePacked(eltTy)) {
          res = evmB.genPackedStorageAddr(baseSlot, arrayIdx, eltTy);
        } else {
          // Slot-aligned layout.
          Value stride = bExt.genI256Const(sol::getStorageSlotCount(eltTy));
          Value scaledIdx = r.create<yul::MulOp>(loc, arrayIdx, stride);
          res = r.create<yul::AddOp>(loc, baseSlot, scaledIdx);
        }

      } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
        // Field index is always constant for struct access.
        auto constIdx = cast<yul::ConstantOp>(idx.getDefiningOp());
        uint64_t fieldIdx = constIdx.getValue().getZExtValue();

        ArrayRef<Type> memberTypes = structTy.getMemberTypes();
        assert(fieldIdx < memberTypes.size() &&
               "struct field index out of range");
        Type fieldTy = memberTypes[fieldIdx];
        auto [slotOffset, byteOffset] =
            structTy.getStorageMemberOffset(fieldIdx);

        // Position for target field.
        if (sol::canBePacked(fieldTy)) {
          // Result: {baseSlot + slotOffset, byteOffset}
          Value slot = r.create<yul::AddOp>(loc, remappedBaseAddr,
                                            bExt.genI256Const(slotOffset));
          res = bExt.genLLVMStruct({slot, bExt.genI256Const(byteOffset)});
        } else {
          res = r.create<yul::AddOp>(loc, remappedBaseAddr,
                                     bExt.genI256Const(slotOffset));
        }

      } else if (isa<sol::StringType>(baseAddrTy)) {
        res = evmB.genStringItemAddress(remappedBaseAddr, getCleanedIdx(), loc);
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

        // Don't generate out-of-bounds check for constant indexing of static
        // arrays.
        if (!isa<BlockArgument>(idx) &&
            isa<yul::ConstantOp>(idx.getDefiningOp()) && !arrTy.isDynSized()) {
          auto constIdx = cast<yul::ConstantOp>(idx.getDefiningOp())
                              .getValue()
                              .getZExtValue();
          // FIXME: Should this be done by the verifier?
          assert(constIdx < static_cast<uint64_t>(arrTy.getSize()));
          unsigned stride = evm::getArrayEltStride(arrTy);
          addrAtIdx = r.create<yul::AddOp>(
              loc, remappedBaseAddr, bExt.genI256Const(constIdx * stride));
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
          auto panicCond =
              bExt.genCmp(yul::CmpPredicate::uge, getCleanedIdx(), size);
          evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

          //
          // Generate the address.
          //
          // In memory every element occupies one 32-byte slot: value types are
          // stored directly, reference types (arrays, structs) are stored as
          // pointers.  In calldata the stride is the ABI head size of the
          // element type.
          Value stride = bExt.genI256Const(evm::getArrayEltStride(arrTy));
          Value scaledIdx = r.create<yul::MulOp>(loc, getCleanedIdx(), stride);
          if (arrTy.isDynSized()) {
            Value dataAddr = evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy);
            addrAtIdx = r.create<yul::AddOp>(loc, dataAddr, scaledIdx);
          } else {
            addrAtIdx = r.create<yul::AddOp>(loc, remappedBaseAddr, scaledIdx);
          }
        }
        assert(addrAtIdx);

        // For calldata pointers to inner elements that have dynamic content,
        // resolve the ABI relative-offset word in the head area.
        Type eltTy = arrTy.getEltType();
        if (dataLoc == sol::DataLocation::CallData &&
            sol::hasDynamicallySizedElt(eltTy)) {
          Value outerDataBase =
              arrTy.isDynSized()
                  ? evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy)
                  : remappedBaseAddr;
          addrAtIdx = evmB.genCalldataAccessRef(eltTy, outerDataBase, addrAtIdx,
                                                /*isNonABI=*/true, loc);
        }

        res = addrAtIdx;

        // Memory/calldata struct
      } else if (auto structTy = dyn_cast<sol::StructType>(baseAddrTy)) {
        auto idxConstOp = cast<yul::ConstantOp>(idx.getDefiningOp());
        uint64_t fieldIdx = idxConstOp.getValue().getZExtValue();

        // In memory every member occupies exactly 32 bytes in the head area.
        // In calldata, members occupy their ABI head size: fixed-length arrays
        // are inline only when their elements are statically sized, while
        // dynamically encoded members contribute a single 32-byte offset. So
        // we must accumulate getCallDataHeadSize() over preceding members
        // rather than multiplying fieldIdx by 32.
        unsigned byteOffset = 0;
        ArrayRef<Type> memberTypes = structTy.getMemberTypes();
        assert(fieldIdx < memberTypes.size() &&
               "struct field index out of range");
        if (dataLoc == sol::DataLocation::CallData) {
          for (uint64_t i = 0; i < fieldIdx; ++i)
            byteOffset += evm::getCallDataHeadSize(memberTypes[i]);
        } else {
          byteOffset = fieldIdx * 32;
        }
        res = r.create<yul::AddOp>(loc, remappedBaseAddr,
                                   bExt.genI256Const(byteOffset));

        // For calldata structs, resolve ABI relative-offset for dynamic
        // members.
        if (dataLoc == sol::DataLocation::CallData) {
          Type memberTy = memberTypes[fieldIdx];
          if (sol::hasDynamicallySizedElt(memberTy))
            res = evmB.genCalldataAccessRef(memberTy, remappedBaseAddr, res,
                                            /*isNonABI=*/true, loc);
        }
        // Bytes (!sol.string)
      } else if (auto strTy = dyn_cast<sol::StringType>(baseAddrTy)) {
        Value size = evmB.genDynSize(remappedBaseAddr, baseAddrTy);
        auto panicCond =
            bExt.genCmp(yul::CmpPredicate::uge, getCleanedIdx(), size);
        evmB.genPanic(mlir::evm::PanicCode::ArrayOutOfBounds, panicCond);

        Value dataAddr = evmB.genDataAddrPtr(remappedBaseAddr, baseAddrTy);
        res = r.create<yul::AddOp>(loc, dataAddr, getCleanedIdx());
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
    evm::Builder evmB(getModule(op), r, loc);

    // Assert that the mapping is a slot (result of sol.addr_of or sol.map).
    assert(cast<IntegerType>(adaptor.getMapping().getType()).getWidth() == 256);

    Type keyTy = op.getKey().getType();
    Value slot;

    if (isa<sol::StringType>(keyTy)) {
      // Dynamic key types (string, bytes): hash = keccak256(abi.encodePacked(
      // rawBytes, slot))
      Value pos = evmB.genFreePtr(loc);
      Value dataLen =
          evmB.genCopyStringDataToMemory(adaptor.getKey(), keyTy, pos, loc);
      Value slotAddr = r.create<yul::AddOp>(loc, pos, dataLen);
      r.create<yul::MStoreOp>(loc, slotAddr, adaptor.getMapping());
      Value totalLen =
          r.create<yul::AddOp>(loc, dataLen, bExt.genI256Const(0x20));
      slot = r.create<yul::Keccak256Op>(loc, pos, totalLen);
    } else {
      // Static key types (integers, address, contract, enum, fixedbytes, byte,
      // ext func ref): value-type key left/right-aligned in 32 bytes, followed
      // by slot. hash = keccak256(key_word, slot_word).
      auto zero = bExt.genI256Const(0);
      Value key = evmB.genCleanup(keyTy, adaptor.getKey(), loc);
      r.create<yul::MStoreOp>(loc, zero, key);
      r.create<yul::MStoreOp>(loc, bExt.genI256Const(0x20),
                              adaptor.getMapping());
      slot = r.create<yul::Keccak256Op>(loc, zero, bExt.genI256Const(0x40));
    }

    // Result is {slot, 0} or just slot depending on pointee type.
    Type resTy = op.getResult().getType();
    if (auto ptrTy = dyn_cast<sol::PointerType>(resTy);
        ptrTy && ptrTy.getDataLocation() == sol::DataLocation::Storage &&
        sol::canBePacked(ptrTy.getPointeeType())) {
      r.replaceOp(op, bExt.genLLVMStruct({slot, bExt.genI256Const(0)}));
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
    evm::Builder evmB(getModule(op), r, loc);

    Value addr = adaptor.getAddr();
    sol::DataLocation dataLoc = sol::getDataLocation(op.getAddr().getType());
    auto i256Ty = r.getIntegerType(256);
    Type eltTy =
        cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();

    switch (dataLoc) {
    case sol::DataLocation::Stack: {
      Type loadTy = getTypeConverter()->convertType(eltTy);
      assert(loadTy);
      r.replaceOpWithNewOp<LLVM::LoadOp>(op, loadTy, addr,
                                         evm::getAlignment(addr));
      return success();
    }
    case sol::DataLocation::Immutable: {
      r.replaceOp(op, evmB.genLoad(addr, dataLoc));
      return success();
    }
    case sol::DataLocation::CallData: {
      auto ld = evmB.genLoad(addr, dataLoc);
      if (isa<IntegerType>(eltTy) || isa<sol::EnumType>(eltTy) ||
          sol::isAddressLikeType(eltTy) || sol::isBytesLikeType(eltTy) ||
          isa<sol::ExtFuncRefType>(eltTy))
        ld = evmB.genCleanup(eltTy, ld, loc, dataLoc);

      r.replaceOp(op, ld);
      return success();
    }
    case sol::DataLocation::Memory: {
      auto ld = evmB.genLoad(addr, dataLoc);
      // old codegen begin
      // Only do cleanup for byte type. This matches the old codegen.
      // (Via-IR cleans every memory load).
      if (isa<sol::ByteType>(eltTy))
        ld = evmB.genCleanup(eltTy, ld, loc, dataLoc);
      // old codegen end

      r.replaceOp(op, ld);
      return success();
    }
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient: {
      // Helper to emit sload or tload based on data location.
      auto genSlotLoad = [&](Value slot) -> Value {
        if (dataLoc == sol::DataLocation::Transient)
          return r.create<yul::TLoadOp>(loc, slot);
        return r.create<yul::SLoadOp>(loc, slot);
      };

      if (sol::canBePacked(eltTy)) {
        // addr is {slot, offset} struct
        Value slot = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, addr, r.getDenseI64ArrayAttr({0}));
        Value offset = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, addr, r.getDenseI64ArrayAttr({1}));

        // Load the full slot and shift right by (offset * 8) to extract.
        Value slotVal = genSlotLoad(slot);
        Value shiftBits =
            r.create<yul::MulOp>(loc, offset, bExt.genI256Const(8));
        Value shifted = r.create<yul::ArithShrOp>(loc, slotVal, shiftBits);
        Value cleaned = evmB.genCleanupPackedStorageValue(eltTy, shifted, loc);
        r.replaceOp(op, cleaned);
      } else {
        // addr is just slot
        Value slotVal = genSlotLoad(addr);
        r.replaceOp(op, slotVal);
      }
      return success();
    }
    };
  }
};

struct LoadImmutableMetadataConversion
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

struct LoadImmutableToYulLowering
    : public OpConversionPattern<sol::LoadImmutableOp> {
  using OpConversionPattern<sol::LoadImmutableOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::LoadImmutableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    bool inRuntime = op->getParentOfType<sol::FuncOp>().getRuntime();
    Value repl;
    if (inRuntime) {
      repl = r.create<yul::LoadImmutableOp>(loc, op.getName());
    } else {
      assert(op->hasAttr("addr"));
      IntegerAttr addr = cast<IntegerAttr>(op->getAttr("addr"));
      repl = r.create<yul::MLoadOp>(loc, bExt.genI256Const(addr.getValue()));
    }

    r.replaceOp(op, repl);
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<sol::StoreOp> {
  using OpConversionPattern<sol::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    Value remappedVal = adaptor.getVal();
    Value remappedAddr = adaptor.getAddr();
    sol::DataLocation dataLoc = sol::getDataLocation(op.getAddr().getType());
    Type eltTy =
        cast<sol::PointerType>(op.getAddr().getType()).getPointeeType();

    switch (dataLoc) {
    case sol::DataLocation::Stack:
      r.replaceOpWithNewOp<LLVM::StoreOp>(op, remappedVal, remappedAddr,
                                          evm::getAlignment(remappedAddr));
      return success();
    case sol::DataLocation::Immutable:
    case sol::DataLocation::Memory: {
      // Generate mstore8 for storing to `bytes`.
      if (isa<sol::ByteType>(eltTy) && dataLoc == sol::DataLocation::Memory) {
        auto byteVal =
            r.create<yul::ByteOp>(loc, bExt.genI256Const(0), remappedVal);
        r.replaceOpWithNewOp<yul::MStore8Op>(op, remappedAddr, byteVal);
        return success();
      }

      if (isa<IntegerType>(eltTy) || isa<sol::EnumType>(eltTy) ||
          sol::isAddressLikeType(eltTy) || isa<sol::FixedBytesType>(eltTy) ||
          isa<sol::ExtFuncRefType>(eltTy))
        remappedVal = evmB.genCleanup(eltTy, remappedVal, loc);

      evmB.genStore(remappedVal, remappedAddr, dataLoc);
      r.eraseOp(op);
      return success();
    }
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient: {
      auto i256Ty = r.getIntegerType(256);

      // Helper to create sstore or tstore based on data location.
      auto genSlotStore = [&](Value slot, Value val) {
        if (dataLoc == sol::DataLocation::Transient)
          r.create<yul::TStoreOp>(loc, slot, val);
        else
          r.create<yul::SStoreOp>(loc, slot, val);
      };

      if (sol::canBePacked(eltTy)) {
        // addr is {slot, offset} struct
        Value slot = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, remappedAddr, r.getDenseI64ArrayAttr({0}));
        Value offset = r.create<LLVM::ExtractValueOp>(
            loc, i256Ty, remappedAddr, r.getDenseI64ArrayAttr({1}));

        // Cleanup the value to be stored.
        Value preparedVal = evmB.genCleanup(eltTy, remappedVal, loc);
        unsigned numBits;
        if (sol::isBytesLikeType(eltTy)) {
          // Bytes-like types: shr to convert from MSB-aligned to LSB-aligned.
          numBits = sol::getNumBytes(eltTy) * 8;
          preparedVal = r.create<yul::ArithShrOp>(
              loc, preparedVal, bExt.genI256Const(256 - numBits));
        } else if (sol::isAddressLikeType(eltTy)) {
          numBits = 160;
        } else if (auto intTy = dyn_cast<IntegerType>(eltTy)) {
          // In storage packing, bool occupies 1 byte not a single bit.
          numBits = intTy.getWidth() == 1 ? 8 : intTy.getWidth();

          // Signed integer cleanup sign-extends to i256. Packed storage stores
          // only the field slice, so drop the sign-extended bits before shl/or.
          if (intTy.isSigned() && numBits < 256)
            preparedVal = r.create<yul::AndOp>(
                loc, preparedVal,
                bExt.genI256Const(APInt::getLowBitsSet(256, numBits)));
        } else if (isa<sol::FuncRefType>(eltTy)) {
          // FuncRef is 64 bits, already i256.
          numBits = 64;
        } else if (isa<sol::ExtFuncRefType>(eltTy)) {
          // ExtFuncRef is MSB-aligned like bytes24. shr(64) to right-align.
          numBits = 192;
          preparedVal = r.create<yul::ArithShrOp>(loc, preparedVal,
                                                  bExt.genI256Const(64));
        } else if (isa<sol::EnumType>(eltTy)) {
          // Enums can have at most 256 members, so always 1 byte.
          numBits = 8;
        } else {
          llvm_unreachable("NYI");
        }

        // Punch hole in slot for new value.
        Value shiftBits =
            r.create<yul::MulOp>(loc, offset, bExt.genI256Const(8));
        Value slotWithHole =
            evmB.genPunchHole(slot, shiftBits, numBits, dataLoc);

        // Shift new value to position: shl(offset * 8, preparedVal)
        preparedVal = r.create<yul::ArithShlOp>(loc, preparedVal, shiftBits);

        // Combine, i.e. or(slotWithHole, preparedVal) and store.
        genSlotStore(slot, r.create<yul::OrOp>(loc, slotWithHole, preparedVal));
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

  // srcDataLoc is passed explicitly because scalar element types (e.g. ui256)
  // carry no data-location annotation and sol::getDataLocation would return
  // stack for them. The top-level caller derives it from the array type, all
  // recursive calls propagate it directly.
  Value genAllocateAndCopy(ModuleOp mod, Value srcAddr, Type ty,
                           sol::DataLocation srcDataLoc, PatternRewriter &r,
                           Location loc) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(mod, r, loc);

    sol::DataLocation dstDataLoc = sol::DataLocation::Memory;

    if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
      Value size, dstAddr, dstDataAddr, srcDataAddr;
      if (arrTy.isDynSized()) {
        size = evmB.genDynSize(srcAddr, ty);
        dstAddr = evmB.genMemAllocForDynArray(
            size, r.create<yul::MulOp>(loc, size, bExt.genI256Const(32)));
        dstDataAddr = evmB.genDataAddrPtr(dstAddr, dstDataLoc);
        srcDataAddr = evmB.genDataAddrPtr(srcAddr, srcDataLoc);
      } else {
        size = bExt.genI256Const(arrTy.getSize());
        dstAddr = evmB.genMemAlloc(arrTy.getSize() * 32);
        dstDataAddr = dstAddr;
        srcDataAddr = srcAddr;
      }

      // Determine element data location: inner array types carry it explicitly,
      // scalar elements inherit from the parent array.
      Type eltTy = arrTy.getEltType();
      sol::DataLocation eltSrcDataLoc = sol::getDataLocation(eltTy);
      if (eltSrcDataLoc == sol::DataLocation::Stack)
        eltSrcDataLoc = srcDataLoc;

      // Packed element types in storage require a specialized extraction loop
      // that reads multiple elements per slot. The generic element-by-element
      // loop below cannot handle packed {slot, byteOffset} addressing.
      if (sol::canBePacked(eltTy) && sol::getNumBytes(eltTy) <= 16 &&
          srcDataLoc == sol::DataLocation::Storage) {
        evmB.genCopy(arrTy, arrTy, srcAddr, dstAddr, srcDataLoc, dstDataLoc,
                     loc);
        return dstAddr;
      }

      // Non-reference value elements in calldata are laid out contiguously, one
      // ABI word (32 bytes) per element, matching the memory layout exactly.
      // Use calldatacopy to copy the entire payload in one operation.
      // TODO: Support multi-dimensional arrays.
      if (srcDataLoc == sol::DataLocation::CallData &&
          sol::canBePacked(eltTy) && (sol::getNumBytes(eltTy) == 32)) {
        Value sizeInBytes = r.create<yul::MulOp>(
            loc, size, bExt.genI256Const(evm::getCallDataHeadSize(eltTy)));
        r.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr,
                                      sizeInBytes);
        return dstAddr;
      }

      bExt.createCountedLoop(
          bExt.genI256Const(0), size, bExt.genI256Const(1), ValueRange{},
          [&](OpBuilder &b, Location loc, Value i256IndVar, ValueRange) {
            Value srcAddrI = evmB.genAddrAtIdx(srcDataAddr, i256IndVar, arrTy,
                                               srcDataLoc, loc);
            Value dstAddrI = evmB.genAddrAtIdx(dstDataAddr, i256IndVar, arrTy,
                                               dstDataLoc, loc);
            // For calldata sources with dynamic inner elements, resolve the ABI
            // relative-offset word in the head area.
            if (srcDataLoc == sol::DataLocation::CallData &&
                sol::hasDynamicallySizedElt(eltTy))
              srcAddrI = evmB.genCalldataAccessRef(eltTy, srcDataAddr, srcAddrI,
                                                   /*isNonABI=*/true, loc);
            Value subElm =
                genAllocateAndCopy(mod, srcAddrI, eltTy, eltSrcDataLoc, r, loc);
            evmB.genStore(subElm, dstAddrI, dstDataLoc);
            return SmallVector<Value>{};
          });
      return dstAddr;
    }

    if (isa<sol::StringType>(ty)) {
      if (srcDataLoc == sol::DataLocation::CallData) {
        Value sizeInBytes = evmB.genDynSize(srcAddr, ty);
        Value memAddr = evmB.genMemAllocForDynArray(
            sizeInBytes, bExt.genRoundUpToMultiple<32>(sizeInBytes));
        Value srcDataAddr = evmB.genDataAddrPtr(srcAddr, srcDataLoc, loc);
        Value dstDataAddr = evmB.genDataAddrPtr(memAddr, dstDataLoc, loc);
        r.create<yul::CallDataCopyOp>(loc, dstDataAddr, srcDataAddr,
                                      sizeInBytes);
        return memAddr;
      }
      if (srcDataLoc == sol::DataLocation::Storage) {
        Value sizeSlot = evmB.genLoad(srcAddr, srcDataLoc, loc);
        Value size = evmB.genStorageStringLength(sizeSlot, loc);
        Value dstAddr = evmB.genMemAllocForDynArray(
            size, bExt.genRoundUpToMultiple<32>(size));
        Value dstDataAddr =
            evmB.genDataAddrPtr(dstAddr, sol::DataLocation::Memory, loc);
        evmB.genStore(size, dstAddr, sol::DataLocation::Memory, loc);
        evmB.genCopyStringDataFromStorageToMemory(srcAddr, sizeSlot, size,
                                                  dstDataAddr, loc);
        return dstAddr;
      }
      llvm_unreachable("NYI: StringType source data location");
    }

    if (auto structTy = dyn_cast<sol::StructType>(ty)) {
      ArrayRef<Type> memberTypes = structTy.getMemberTypes();
      unsigned numMembers = memberTypes.size();

      // Allocate head area: one 32-byte slot per member.
      Value dstBase = evmB.genMemAlloc(numMembers * 32);
      auto reader = evmB.makeStructMemberReader(structTy, r, loc, srcAddr);

      for (uint64_t i = 0; i < numMembers; ++i) {
        Type memberTy = memberTypes[i];
        Value dstHeadSlot =
            r.create<yul::AddOp>(loc, dstBase, bExt.genI256Const(i * 32));
        Value srcResult = reader->read(i);

        if (sol::isNonPtrRefType(memberTy)) {
          Value memPtr = genAllocateAndCopy(
              mod, srcResult, memberTy, sol::getDataLocation(memberTy), r, loc);
          evmB.genStore(memPtr, dstHeadSlot, sol::DataLocation::Memory);
        } else {
          Value cleanedResult =
              evmB.genCleanup(memberTy, srcResult, loc, srcDataLoc);
          evmB.genStore(cleanedResult, dstHeadSlot, sol::DataLocation::Memory);
        }
        if (i + 1 < numMembers)
          reader->advance(i);
      }
      return dstBase;
    }

    assert(sol::canBePacked(ty));
    Value val = evmB.genLoad(srcAddr, srcDataLoc);
    unsigned shift = 0;
    if (srcDataLoc == sol::DataLocation::Storage) {
      if (isa<sol::ExtFuncRefType>(ty))
        shift = 64;
      else if (sol::isBytesLikeType(ty))
        shift = (32 - sol::getNumBytes(ty)) * 8;
    }
    if (shift > 0)
      val = r.create<yul::ArithShlOp>(loc, val, bExt.genI256Const(shift));
    val = evmB.genCleanup(ty, val, loc, srcDataLoc);
    return val;
  }

  LogicalResult matchAndRewrite(sol::DataLocCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    Type srcTy = op.getInp().getType();
    Type dstTy = op.getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);
    assert(!isa<sol::MappingType>(srcTy));

    if (dstDataLoc == sol::DataLocation::Memory &&
        sol::isNonPtrRefType(srcTy)) {
      r.replaceOp(op, genAllocateAndCopy(getModule(op), adaptor.getInp(), srcTy,
                                         srcDataLoc, r, loc));
      return success();
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
    evm::Builder evmB(getModule(op), r, loc);

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
    evm::Builder evmB(getModule(op), r, loc);

    Type arrTy = op.getArr().getType();
    assert(sol::getDataLocation(arrTy) == sol::DataLocation::CallData);

    Value arr = adaptor.getArr();
    Value start =
        evmB.genCleanup(op.getStart().getType(), adaptor.getStart(), loc);
    Value end = evmB.genCleanup(op.getEnd().getType(), adaptor.getEnd(), loc);

    Value dataAddr = evmB.genDataAddrPtr(arr, arrTy);
    Value length = evmB.genDynSize(arr, arrTy);

    // Validate: start <= end
    auto startGtEnd = bExt.genCmp(yul::CmpPredicate::ugt, start, end);
    evmB.genDebugRevertWithMsg(startGtEnd, "Slice starts after end", loc);

    // Validate: end <= length
    auto endGtLen = bExt.genCmp(yul::CmpPredicate::ugt, end, length);
    evmB.genDebugRevertWithMsg(endGtLen, "Slice is greater than length", loc);

    // Compute stride (element size in calldata)
    unsigned stride = 32;
    if (auto arrType = dyn_cast<sol::ArrayType>(arrTy))
      stride = evm::getCallDataHeadSize(arrType.getEltType());
    else if (isa<sol::StringType>(arrTy))
      stride = 1;

    // newOffset = dataAddr + start * stride
    Value scaledStart =
        r.create<yul::MulOp>(loc, start, bExt.genI256Const(stride));
    Value newOffset = r.create<yul::AddOp>(loc, dataAddr, scaledStart);

    // newLength = end - start
    Value newLength = r.create<yul::SubOp>(loc, end, start);

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
    evm::Builder evmB(getModule(op), r, loc);

    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    sol::DataLocation srcDataLoc = sol::getDataLocation(srcTy);
    sol::DataLocation dstDataLoc = sol::getDataLocation(dstTy);

    evmB.genCopy(srcTy, dstTy, adaptor.getSrc(), adaptor.getDst(), srcDataLoc,
                 dstDataLoc, loc);
    r.eraseOp(op);
    return success();
  }
};

struct DeleteOpLowering : public OpConversionPattern<sol::DeleteOp> {
  using OpConversionPattern<sol::DeleteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::DeleteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    // The operand is a storage reference: `address_type` makes the aggregate
    // type itself the reference type, and AddrOfOp lowers a non-packable storage
    // variable to its i256 slot. So the original operand type is the value type
    // to clear, and the remapped operand is the slot.
    Type refTy = op.getReference().getType();
    assert(sol::getDataLocation(refTy) == sol::DataLocation::Storage &&
           "sol.delete expects a Storage reference");
    evmB.genClearStorageValue(refTy, adaptor.getReference(), loc);
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
    r.replaceOp(op, r.create<yul::AndOp>(loc, data, mask));
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

struct BlockHashOpLowering : public OpConversionPattern<sol::BlockHashOp> {
  using OpConversionPattern<sol::BlockHashOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(sol::BlockHashOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::BlockHashOp>(op, adaptor.getBlockNumber());
    return success();
  }
};

struct BlobHashOpLowering : public OpConversionPattern<sol::BlobHashOp> {
  using OpConversionPattern<sol::BlobHashOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(sol::BlobHashOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    r.replaceOpWithNewOp<yul::BlobHashOp>(op, adaptor.getIdx());
    return success();
  }
};

struct SelfdestructOpLowering
    : public CleanedOperandsLowering<sol::SelfdestructOp,
                                     SelfdestructOpLowering> {
  using Base =
      CleanedOperandsLowering<sol::SelfdestructOp, SelfdestructOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::SelfdestructOp op,
                               ConversionPatternRewriter &r,
                               Value recipient) const {
    r.replaceOpWithNewOp<yul::SelfDestructOp>(op, recipient);
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

// old codegen begin
// (Via-IR cleans on casts, not here).
struct CodeHashOpLowering
    : public CleanedOperandsLowering<sol::CodeHashOp, CodeHashOpLowering> {
  using Base = CleanedOperandsLowering<sol::CodeHashOp, CodeHashOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CodeHashOp op, ConversionPatternRewriter &r,
                               Value addr) const {
    r.replaceOpWithNewOp<yul::ExtCodeHashOp>(op, addr);
    return success();
  }
};

struct BalanceOpLowering
    : public CleanedOperandsLowering<sol::BalanceOp, BalanceOpLowering> {
  using Base = CleanedOperandsLowering<sol::BalanceOp, BalanceOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::BalanceOp op, ConversionPatternRewriter &r,
                               Value addr) const {
    r.replaceOpWithNewOp<yul::BalanceOp>(op, addr);
    return success();
  }
};
// old codegen end

struct EncodeOpLowering : public OpConversionPattern<sol::EncodeOp> {
  using OpConversionPattern<sol::EncodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::EncodeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    Value freePtr = evmB.genFreePtr();
    if (op.getPacked()) {
      Value dataStart =
          r.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(32));
      Value dataEnd = evmB.genABIEncodingPacked(op.getIns().getType(),
                                                adaptor.getIns(), dataStart);
      Value dataSize = r.create<yul::SubOp>(loc, dataEnd, dataStart);
      r.create<yul::MStoreOp>(loc, freePtr, dataSize);
      Value allocationSize = r.create<yul::SubOp>(loc, dataEnd, freePtr);
      evmB.genFreePtrUpd(freePtr, allocationSize);
    } else if (op.getSelector()) {
      assert(adaptor.getSelector() && "selector operand is required");

      Value selectorAddr =
          r.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(32));
      r.create<yul::MStoreOp>(loc, selectorAddr, adaptor.getSelector());

      Value tupleStart =
          r.create<yul::AddOp>(loc, selectorAddr, bExt.genI256Const(4));
      Value tupleEnd = evmB.genABIEncoding(op.getIns().getType(),
                                           adaptor.getIns(), tupleStart);
      Value dataSize = r.create<yul::SubOp>(loc, tupleEnd, selectorAddr);
      r.create<yul::MStoreOp>(loc, freePtr, dataSize);
      Value allocationSize = r.create<yul::SubOp>(loc, tupleEnd, freePtr);
      evmB.genFreePtrUpd(freePtr, allocationSize);
    } else {
      Value tupleStart =
          r.create<yul::AddOp>(loc, freePtr, bExt.genI256Const(32));
      Value tupleEnd = evmB.genABIEncoding(op.getIns().getType(),
                                           adaptor.getIns(), tupleStart);
      Value tupleSize = r.create<yul::SubOp>(loc, tupleEnd, tupleStart);
      r.create<yul::MStoreOp>(loc, freePtr, tupleSize);
      Value allocationSize = r.create<yul::SubOp>(loc, tupleEnd, freePtr);
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
    evm::Builder evmB(getModule(op), r, loc);

    std::vector<Value> results;
    Type srcTy = op.getAddr().getType();
    auto srcStringTy = dyn_cast<sol::StringType>(srcTy);
    assert(srcStringTy &&
           "abi.decode source must be bytes memory/calldata in sol dialect");
    sol::DataLocation dataLoc = srcStringTy.getDataLocation();
    assert((dataLoc == sol::DataLocation::Memory ||
            dataLoc == sol::DataLocation::CallData) &&
           "abi.decode expects memory/calldata byte buffer");

    Value tupleStart = evmB.genDataAddrPtr(adaptor.getAddr(), srcTy);
    Value tupleSize = evmB.genDynSize(adaptor.getAddr(), srcTy);
    Value tupleEnd = r.create<yul::AddOp>(loc, tupleStart, tupleSize);
    bool fromMem = dataLoc == sol::DataLocation::Memory;
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
    auto mod = getModule(op);
    evm::Builder evmB(mod, r, loc);

    assert(sol::evmCanOverchargeGasForCall(mod) && "NYI");

    // TODO:
    // - The return arg analysis is done for evm's the supports return data (See
    // solidity::frontend::ReturnInfo).
    // - The generated code for the return has returndatacopy and returnsize.
    assert(sol::evmSupportsReturnData(mod) && "NYI");

    SmallVector<Type> abiInputTys =
        getABITypes(r, op.getIns().getTypes(), op.getCalleeType().getInputs(),
                    op.getLibraryCall());
    SmallVector<Type> abiResultTys =
        getABITypes(r, op.getCalleeType().getResults(),
                    op.getCalleeType().getResults(), op.getLibraryCall());

    // Check if we need to generate the extcodesize check.
    unsigned totHeadSize = 0;
    for (auto resTy : abiResultTys) {
      totHeadSize += evm::getCallDataHeadSize(resTy);
    }

    bool extCodeSizeCheck = totHeadSize == 0 ||
                            !sol::evmSupportsReturnData(mod) ||
                            sol::shouldEmitDebugRevertStrings(mod);

    Value addr = adaptor.getAddr();
    // old codegen begin
    // (Via-IR cleans on casts, not here).
    if (isa<sol::AddressType>(op.getAddr().getType()))
      addr = evmB.genCleanup(op.getAddr().getType(), addr, loc);
    // old codegen end

    if (extCodeSizeCheck) {
      // Generate the revert code.
      auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, addr);
      auto isExtCodeSizeZero =
          bExt.genCmp(yul::CmpPredicate::eq, extCodeSize, bExt.genI256Const(0));
      evmB.genDebugRevertWithMsg(isExtCodeSizeZero,
                                 "Target contract does not contain code");
    }

    // Generate the store of the selector.
    Value selectorAddr = evmB.genFreePtr();
    Value shiftedSelector = r.create<yul::ArithShlOp>(
        loc, adaptor.getSelector(), bExt.genI256Const(224));
    r.create<yul::MStoreOp>(loc, selectorAddr, shiftedSelector);

    // Generate the abi encoding code.
    Value tupleStart =
        r.create<yul::AddOp>(loc, selectorAddr, bExt.genI256Const(4));
    Value tupleEnd =
        evmB.genABIEncoding(abiInputTys, adaptor.getIns(), tupleStart);

    // Calculate the return size statically and/or check if it's dynamic. This
    // is copied from solidity::frontend::ReturnInfo.
    unsigned staticRetSizeVal = 0;
    bool isRetSizeDynamic = false;
    for (Type ty : abiResultTys) {
      if (sol::isDynamicallySized(ty)) {
        isRetSizeDynamic = true;
        staticRetSizeVal = 0;
        break;
      }
      staticRetSizeVal += evm::getCallDataHeadSize(ty);
    }

    // Generate the call.
    Value inpSize = r.create<yul::SubOp>(loc, tupleEnd, selectorAddr);
    Value staticRetSize = bExt.genI256Const(staticRetSizeVal);
    mlir::Value status;

    // Order is important here, staticcall might overlap with delegatecall.
    if (op.getDelegateCall())
      status = r.create<yul::DelegateCallOp>(
          loc, adaptor.getGas(), addr,
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);
    else if (op.getStaticCall())
      status = r.create<yul::StaticCallOp>(loc, adaptor.getGas(), addr,
                                           /*inpOffset=*/selectorAddr, inpSize,
                                           /*outOffset=*/selectorAddr,
                                           /*outSize=*/staticRetSize);
    else
      status = r.create<yul::CallOp>(
          loc, adaptor.getGas(), addr, adaptor.getVal(),
          /*inpOffset=*/selectorAddr, inpSize,
          /*outOffset=*/selectorAddr, /*outSize=*/staticRetSize);

    // Generate forwarding revert if not try-call.
    if (!op.getTryCall()) {
      auto statusIsZero =
          bExt.genCmp(yul::CmpPredicate::eq, status, bExt.genI256Const(0));
      evmB.genForwardingRevert(statusIsZero);
    }

    // Get the types of the results from the decoding, which should be the same
    // as the corsp legal types.
    SmallVector<Type> decodedResultTys;
    if (failed(getTypeConverter()->convertTypes(op.getCalleeType().getResults(),
                                                decodedResultTys)))
      return failure();

    // Generate the if-else op of the status that yields the decoded results.
    auto statusIsNotZero =
        bExt.genCmp(yul::CmpPredicate::ne, status, bExt.genI256Const(0));
    auto statusIfOp = r.create<yul::IfOp>(loc, /*resultTys=*/decodedResultTys,
                                          /*cond=*/statusIsNotZero);

    // Generate the else block (failure) which yields undefs. The undefs will
    // not be used during execution as the either control flow will skip the
    // sol.try's success block or hit the previous forwarding revert.
    r.setInsertionPointToStart(&statusIfOp.getElseRegion().emplaceBlock());
    SmallVector<Value, 2> undefYields;
    for (Type ty : decodedResultTys) {
      undefYields.push_back(r.create<LLVM::UndefOp>(loc, ty));
    }
    r.create<yul::YieldOp>(loc, undefYields);

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
      Value tupleEnd = r.create<yul::AddOp>(loc, selectorAddr, retDataSize);
      evmB.genABITupleDecoding(abiResultTys, selectorAddr, tupleEnd,
                               decodedResults, /*fromMem=*/true);
    } else {
      // See https://github.com/ethereum/solidity/pull/12684
      Value staticRetSizeGreater =
          bExt.genCmp(yul::CmpPredicate::ugt, staticRetSize, retDataSize);

      auto ifOp =
          bExt.createIf(TypeRange{r.getIntegerType(256)}, staticRetSizeGreater);
      // Then block:
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      evmB.genFreePtrUpd(selectorAddr, retDataSize);
      Value tupleEnd = r.create<yul::AddOp>(loc, selectorAddr, retDataSize);
      r.create<yul::YieldOp>(loc, ValueRange{tupleEnd});
      // Else block:
      r.setInsertionPointToStart(&ifOp.getElseRegion().front());
      evmB.genFreePtrUpd(selectorAddr, staticRetSize);
      tupleEnd = r.create<yul::AddOp>(loc, selectorAddr, staticRetSize);
      r.create<yul::YieldOp>(loc, ValueRange{tupleEnd});

      r.setInsertionPointAfter(ifOp);
      evmB.genABITupleDecoding(abiResultTys, selectorAddr, ifOp.getResult(0),
                               decodedResults,
                               /*fromMem=*/true);
    }
    r.create<yul::YieldOp>(loc, decodedResults);

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
  auto mod = getModule(op);
  evm::Builder evmB(mod, r, loc);

  assert(sol::evmSupportsReturnData(mod) && "NYI");

  auto inpTy = cast<sol::StringType>(op.getInp().getType());
  Value inpSize = evmB.genDynSize(adaptor.getInp(), inpTy, loc);
  Value inpStart = evmB.genDataAddrPtr(adaptor.getInp(), inpTy, loc);
  // old codegen begin
  // (Via-IR cleans on casts, not here).
  Value addr = evmB.genCleanup(op.getAddr().getType(), adaptor.getAddr(), loc);
  // old codegen end
  Value zero = bExt.genI256Const(0);
  Value rawStatus =
      opBuilderFunc(loc, adaptor, addr, inpStart, inpSize, zero, r);

  Value status = bExt.genCmp(yul::CmpPredicate::ne, rawStatus, zero);
  Value retDataSize = r.create<yul::ReturnDataSizeOp>(loc);
  Value roundedRetDataSize = bExt.genRoundUpToMultiple<32>(retDataSize);
  Value retData =
      evmB.genMemAllocForDynArray(retDataSize, roundedRetDataSize, loc);
  Value retDataStart =
      evmB.genDataAddrPtr(retData, sol::DataLocation::Memory, loc);
  r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/retDataStart,
                                  /*src=*/zero, retDataSize);

  r.replaceOp(op, SmallVector<Value, 2>{status, retData});
  return success();
}

struct BareCallOpLowering : public OpConversionPattern<sol::BareCallOp> {
  using OpConversionPattern<sol::BareCallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::BareCallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    return lowerBareCallLikeOp(
        op, adaptor, r,
        [](Location loc, auto adaptor, Value addr, Value inpStart,
           Value inpSize, Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::CallOp>(
              loc, adaptor.getGas(), addr, adaptor.getVal(),
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
        [](Location loc, auto adaptor, Value addr, Value inpStart,
           Value inpSize, Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::DelegateCallOp>(loc, adaptor.getGas(), addr,
                                               /*inpOffset=*/inpStart, inpSize,
                                               /*outOffset=*/zero,
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
        [](Location loc, auto adaptor, Value addr, Value inpStart,
           Value inpSize, Value zero, ConversionPatternRewriter &r) {
          return r.create<yul::StaticCallOp>(loc, adaptor.getGas(), addr,
                                             /*inpOffset=*/inpStart, inpSize,
                                             /*outOffset=*/zero,
                                             /*outSize=*/zero);
        });
  }
};

/// Lowers the common value-transfer call pattern used by 'sol.send' and
/// 'sol.transfer' and returns the requested status predicate.
template <typename OpT, typename AdaptorT>
static Value genValueTransferStatus(OpT op, AdaptorT adaptor,
                                    yul::CmpPredicate pred,
                                    ConversionPatternRewriter &r) {
  Location loc = op.getLoc();
  mlir::solgen::BuilderExt bExt(r, loc);
  evm::Builder evmB(getModule(op), r, loc);
  Value zero = bExt.genI256Const(0);
  Value callStipend = bExt.genI256Const(2300);
  Value valueIsZero =
      bExt.genCmp(yul::CmpPredicate::eq, adaptor.getVal(), zero);
  Value gas = r.create<yul::ArithSelectOp>(loc, valueIsZero, callStipend, zero);
  // old codegen begin
  // (Via-IR cleans on casts, not here).
  Value addr = evmB.genCleanup(op.getAddr().getType(), adaptor.getAddr(), loc);
  // old codegen end

  Value callStatus = r.create<yul::CallOp>(loc, gas, addr, adaptor.getVal(),
                                           /*inpOffset=*/zero, /*inpSize=*/zero,
                                           /*outOffset=*/zero,
                                           /*outSize=*/zero);
  return bExt.genCmp(pred, callStatus, zero);
}

struct SendOpLowering : public OpConversionPattern<sol::SendOp> {
  using OpConversionPattern<sol::SendOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::SendOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Value status =
        genValueTransferStatus(op, adaptor, yul::CmpPredicate::ne, r);
    r.replaceOp(op, status);
    return success();
  }
};

struct TransferOpLowering : public OpConversionPattern<sol::TransferOp> {
  using OpConversionPattern<sol::TransferOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::TransferOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    Value statusIsZero =
        genValueTransferStatus(op, adaptor, yul::CmpPredicate::eq, r);
    evmB.genForwardingRevert(statusIsZero);
    r.eraseOp(op);
    return success();
  }
};

struct ExtICallOpLowering : public OpConversionPattern<sol::ExtICallOp> {
  using OpConversionPattern<sol::ExtICallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ExtICallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);

    // Split MSB-aligned ext func ref into addr and selector:
    // selector = (callee >> 64) & 0xffffffff; addr = callee >> 96
    Value callee = adaptor.getCallee();
    Value shifted =
        r.create<yul::ArithShrOp>(loc, callee, bExt.genI256Const(64));
    Value selector = r.create<yul::AndOp>(
        loc, shifted, bExt.genI256Const(APInt::getLowBitsSet(256, 32)));
    Value addr = r.create<yul::ArithShrOp>(loc, shifted, bExt.genI256Const(32));

    // Get callee function type from the ext_func_ref type.
    auto extFuncRefTy = cast<sol::ExtFuncRefType>(op.getCallee().getType());
    auto calleeType = extFuncRefTy.getFuncTy();

    // Lower to ExtCallOp - reuse all its lowering logic.
    // Both ExtICallOp and ExtCallOp return (i1 status, results...).
    auto extCall = r.create<sol::ExtCallOp>(loc, op.getResultTypes(),
                                            /*callee=*/"",
                                            /*ins=*/op.getCalleeOperands(),
                                            /*addr=*/addr,
                                            /*gas=*/adaptor.getGas(),
                                            /*val=*/adaptor.getValue(),
                                            /*selector=*/selector,
                                            /*try_call=*/op.getTryCall(),
                                            /*static_call=*/op.getStaticCall(),
                                            /*delegate_call=*/false,
                                            /*library_call=*/false,
                                            /*callee_type=*/calleeType,
                                            /*arg_attrs=*/ArrayAttr{},
                                            /*res_attrs=*/ArrayAttr{});

    r.replaceOp(op, extCall.getResults());
    return success();
  }
};

struct NewOpLowering : public OpConversionPattern<sol::NewOp> {
  using OpConversionPattern<sol::NewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::NewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    Value bytecodeAddr = evmB.genFreePtr();
    Value dataOffset = r.create<yul::DataOffsetOp>(loc, op.getObjName());
    Value dataSize = r.create<yul::DataSizeOp>(loc, op.getObjName());
    r.create<yul::CodeCopyOp>(loc, bytecodeAddr, dataOffset, dataSize);

    Value tupleStart = r.create<yul::AddOp>(loc, bytecodeAddr, dataSize);
    Value tupleEnd = evmB.genABIEncoding(op.getCtorArgs().getType(),
                                         adaptor.getCtorArgs(), tupleStart);
    Value allocSize = r.create<yul::SubOp>(loc, tupleEnd, bytecodeAddr);

    Value status;
    if (op.getSalt())
      status = r.create<yul::Create2Op>(loc, adaptor.getVal(), bytecodeAddr,
                                        allocSize, adaptor.getSalt());
    else
      status = r.create<yul::CreateOp>(loc, adaptor.getVal(), bytecodeAddr,
                                       allocSize);

    Value addr = status;
    if (!op.getTryCall()) {
      Value zero = bExt.genI256Const(0);
      Value addrIsZero = bExt.genCmp(yul::CmpPredicate::eq, addr, zero);
      evmB.genForwardingRevert(addrIsZero);
    }
    r.replaceOp(op, addr);
    return success();
  }
};

struct ObjectCodeOpLowering : public OpConversionPattern<sol::ObjectCodeOp> {
  using OpConversionPattern<sol::ObjectCodeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ObjectCodeOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    Value dataSize = r.create<yul::DataSizeOp>(loc, op.getObjName());
    Value alloc = evmB.genMemAlloc(op.getType(), /*zeroInit=*/false,
                                   /*initVals=*/{}, dataSize, Type{});
    Value dataAddr = evmB.genDataAddrPtr(alloc, sol::DataLocation::Memory);
    Value dataOffset = r.create<yul::DataOffsetOp>(loc, op.getObjName());
    r.create<yul::CodeCopyOp>(loc, dataAddr, dataOffset, dataSize);
    r.replaceOp(op, alloc);
    return success();
  }
};

// old codegen begin
// (Via-IR cleans on casts, not here).
struct CodeOpLowering
    : public CleanedOperandsLowering<sol::CodeOp, CodeOpLowering> {
  using Base = CleanedOperandsLowering<sol::CodeOp, CodeOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::CodeOp op, ConversionPatternRewriter &r,
                               Value addr) const {

    Location loc = op.getLoc();
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    auto extCodeSize = r.create<yul::ExtCodeSizeOp>(loc, addr);
    Value alloc = evmB.genMemAlloc(op.getType(), /*zeroInit=*/false,
                                   /*initVals=*/{}, extCodeSize, Type{});
    auto codeAddr = evmB.genDataAddrPtr(alloc, sol::DataLocation::Memory);
    r.create<yul::ExtCodeCopyOp>(loc, addr, /*dstOffset=*/codeAddr,
                                 /*srcOffset=*/bExt.genI256Const(0),
                                 extCodeSize);
    r.replaceOp(op, alloc);
    return success();
  }
};
// old codegen end

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
      evm::Builder evmB(getModule(tryOp), r, loc);
      r.setInsertionPointToStart(&ifStatus.getElseRegion().emplaceBlock());
      evmB.genForwardingRevert();
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));
    } else {
      r.inlineRegionBefore(tryOp.getFallbackRegion(), ifStatus.getElseRegion(),
                           ifStatus.getElseRegion().begin());
      // If the fallback clause declared a parameter, materialize the raw return
      // data as a `bytes memory` and replace uses of the block argument.
      Block &entry = ifStatus.getElseRegion().front();
      if (entry.getNumArguments() > 0) {
        BlockArgument blkArg = entry.getArgument(0);
        Location argLoc = blkArg.getLoc();
        OpBuilder::InsertionGuard g(r);
        r.setInsertionPointToStart(&entry);
        evm::Builder evmB(getModule(tryOp), r, argLoc);
        auto memBytesTy =
            sol::StringType::get(r.getContext(), sol::DataLocation::Memory);
        Value retDataSize = r.create<yul::ReturnDataSizeOp>(argLoc);
        Value buf =
            evmB.genMemAlloc(memBytesTy, /*zeroInit=*/false,
                             /*initVals=*/{}, retDataSize, Type{}, argLoc);
        Value dataPtr =
            evmB.genDataAddrPtr(buf, sol::DataLocation::Memory, argLoc);
        r.create<yul::ReturnDataCopyOp>(argLoc, /*dst=*/dataPtr,
                                        /*src=*/bExt.genI256Const(0, argLoc),
                                        retDataSize);
        Value repl = getTypeConverter()->materializeSourceConversion(
            r, argLoc, blkArg.getType(), buf);
        blkArg.replaceAllUsesWith(repl);
        entry.eraseArgument(0);
      }
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
    auto selectorRetCond = bExt.genCmp(yul::CmpPredicate::ugt, returnDataSize,
                                       bExt.genI256Const(3));
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
        r.create<yul::ArithShrOp>(loc, selectorWord, bExt.genI256Const(224));

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
      auto panicRetCond = bExt.genCmp(yul::CmpPredicate::ugt, returnDataSize,
                                      bExt.genI256Const(0x23));
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

      evm::Builder evmB(getModule(tryOp), r, loc);

      // Decode the ABI-encoded Error(string) like upstream's
      // try_decode_error_message. Three nested guards; a failed one leaves
      // runFallbackFlag set so the fallback clause runs:
      //
      // - 0x44 = 4 (selector) + 0x20 (offset) + 0x20 (length)
      // - 0x24 = 4 (selector) + 0x20 (offset)
      // - offset is the ABI offset to the string body, read from the head.
      // - length is the string length, read at msg.
      // - dataEnd is one past the last byte of the returned tuple.
      //
      // clang-format off
      //
      //   if (returndatasize >= 0x44)
      //     if (offset <= u64Max && offset + 0x24 <= returndatasize)
      //       if (length <= u64Max && msg + 0x20 + length <= dataEnd)
      //         run the Error clause
      //
      // clang-format on
      Value sizeOk = bExt.genCmp(yul::CmpPredicate::uge, returnDataSize,
                                 bExt.genI256Const(0x44));
      auto ifSizeOk = r.create<sol::IfOp>(loc, sizeOk);
      r.setInsertionPointToStart(&ifSizeOk.getThenRegion().emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));
      Value u64Max = bExt.genI256Const(APInt::getLowBitsSet(256, 64));
      Value abiTupleSize =
          r.create<yul::SubOp>(loc, returnDataSize, bExt.genI256Const(4));
      Value abiTuple = evmB.genMemAlloc(abiTupleSize);
      r.create<yul::ReturnDataCopyOp>(loc, /*dst=*/abiTuple,
                                      /*src=*/bExt.genI256Const(4),
                                      abiTupleSize);
      Value offset = r.create<yul::MLoadOp>(loc, abiTuple);
      Value offsetSmall = bExt.genCmp(yul::CmpPredicate::ule, offset, u64Max);
      Value offsetEnd =
          r.create<yul::AddOp>(loc, offset, bExt.genI256Const(0x24));
      Value offsetInBounds =
          bExt.genCmp(yul::CmpPredicate::ule, offsetEnd, returnDataSize);

      Value offsetOk = r.create<yul::AndOp>(loc, offsetSmall, offsetInBounds);
      auto ifOffsetOk = r.create<sol::IfOp>(loc, offsetOk);
      r.setInsertionPointToStart(&ifOffsetOk.getThenRegion().emplaceBlock());
      r.setInsertionPoint(r.create<sol::YieldOp>(loc));

      Value errMsg = r.create<yul::AddOp>(loc, abiTuple, offset);
      Value length = r.create<yul::MLoadOp>(loc, errMsg);
      Value lengthSmall = bExt.genCmp(yul::CmpPredicate::ule, length, u64Max);
      Value msgEnd = r.create<yul::AddOp>(
          loc, r.create<yul::AddOp>(loc, errMsg, bExt.genI256Const(0x20)),
          length);
      Value dataEnd = r.create<yul::AddOp>(loc, abiTuple, abiTupleSize);
      Value lengthInBounds =
          bExt.genCmp(yul::CmpPredicate::ule, msgEnd, dataEnd);
      Value lengthOk = r.create<yul::AndOp>(loc, lengthSmall, lengthInBounds);
      auto ifLengthOk = r.create<sol::IfOp>(loc, lengthOk);
      r.inlineRegionBefore(tryOp.getErrorRegion(), ifLengthOk.getThenRegion(),
                           ifLengthOk.getThenRegion().begin());
      Block &thenEntry = ifLengthOk.getThenRegion().front();
      r.setInsertionPointToStart(&thenEntry);
      r.create<sol::StoreOp>(loc, bExt.genBool(false), runFallbackFlag);
      BlockArgument blkArg = thenEntry.getArgument(0);
      auto blkArgRepl = getTypeConverter()->materializeSourceConversion(
          r, loc, blkArg.getType(), errMsg);
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
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(getModule(op), r, loc);

    // Generate: if (!cond) { panic(0x01) }
    mlir::Value falseVal = bExt.genBool(false);
    mlir::Value negCond =
        bExt.genCmp(yul::CmpPredicate::eq, adaptor.getCond(), falseVal);
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
    evm::Builder evmB(getModule(op), r, loc);

    // Generate the revert condition.
    mlir::Value falseVal = bExt.genBool(false);
    mlir::Value negCond =
        bExt.genCmp(yul::CmpPredicate::eq, adaptor.getCond(), falseVal);
    if (op.getCall()) {
      assert(op.getMsg());
      evmB.genRevert(negCond, op.getArgs().getTypes(), adaptor.getArgs(),
                     *op.getMsg());
      r.eraseOp(op);
      return success();
    }

    // Generate the revert.
    if (op.getMsg())
      evmB.genUserRevertWithMsg(negCond, op.getMsg()->str());
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
    evm::Builder evmB(getModule(op), r, loc);

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
    for (Value origIdxArg : op.getIndexedArgs()) {
      Type origTy = origIdxArg.getType();
      Value val = remappedOperands[argIdx++];

      // Reference-type indexed arg: topic = keccak256(packed_encode(arg)).
      if (isa<sol::StringType, sol::ArrayType, sol::StructType>(origTy)) {
        Value scratchStart = evmB.genFreePtr();
        Value scratchEnd = evmB.genABIEncodingPacked(
            TypeRange{origTy}, ValueRange{val}, scratchStart, loc);
        Value scratchSize = r.create<yul::SubOp>(loc, scratchEnd, scratchStart);
        indexedArgs.push_back(
            r.create<yul::Keccak256Op>(loc, scratchStart, scratchSize));
        continue;
      }

      // Value-type indexed arg: cleanup-and-widen to i256 (LogOp expects
      // i256 topics).
      indexedArgs.push_back(evmB.genCleanup(origTy, val, loc));
    }

    for (Value arg : op.getNonIndexedArgs()) {
      nonIndexedArgsType.push_back(arg.getType());
      nonIndexedArgs.push_back(remappedOperands[argIdx++]);
    }

    // Generate the tuple encoding for the non-indexed args.
    // TODO: Are we sure we need an unbounded allocation here?
    Value tupleStart = evmB.genFreePtr();
    Value tupleEnd =
        evmB.genABIEncoding(nonIndexedArgsType, nonIndexedArgs, tupleStart);
    Value tupleSize = r.create<yul::SubOp>(loc, tupleEnd, tupleStart);

    // Generate sol.log and replace sol.emit with it.
    r.replaceOpWithNewOp<yul::LogOp>(op, tupleStart, tupleSize, indexedArgs);

    return success();
  }
};

struct RevertOpLowering : public OpConversionPattern<sol::RevertOp> {
  using OpConversionPattern<sol::RevertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::RevertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    evm::Builder evmB(getModule(op), r, op.getLoc());
    if (op.getCall()) {
      // revert ErrorName(...)
      assert(!op.getSignature().empty());
      evmB.genRevert(op.getArgs().getTypes(), adaptor.getArgs(),
                     op.getSignature());
    } else if (!op.getSignature().empty()) {
      // revert("reason")
      evmB.genUserRevertWithMsg(op.getSignature().str());
    } else {
      // revert()
      evmB.genRevert(op.getLoc());
    }
    r.eraseOp(op);
    return success();
  }
};

static void lowerSolRegionTerminatorsToYul(Region &region,
                                           ConversionPatternRewriter &r) {
  for (Block &block : region) {
    Operation *terminator = block.getTerminator();
    if (auto yieldOp = dyn_cast<sol::YieldOp>(terminator)) {
      assert(yieldOp.getIns().empty() &&
             "Yul structured control flow does not support yielded values yet");
      r.setInsertionPoint(yieldOp);
      r.replaceOpWithNewOp<yul::YieldOp>(yieldOp);
    } else if (auto conditionOp = dyn_cast<sol::ConditionOp>(terminator)) {
      r.setInsertionPoint(conditionOp);
      Value cond = conditionOp.getCondition();
      if (Value remappedCond = r.getRemappedValue(cond))
        cond = remappedCond;
      r.replaceOpWithNewOp<yul::ConditionOp>(conditionOp, cond, ValueRange{});
    } else if (auto breakOp = dyn_cast<sol::BreakOp>(terminator)) {
      r.setInsertionPoint(breakOp);
      r.replaceOpWithNewOp<yul::BreakOp>(breakOp);
    } else if (auto continueOp = dyn_cast<sol::ContinueOp>(terminator)) {
      r.setInsertionPoint(continueOp);
      r.replaceOpWithNewOp<yul::ContinueOp>(continueOp);
    }
  }
}

struct IfOpLowering : public OpConversionPattern<sol::IfOp> {
  using OpConversionPattern<sol::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::IfOp ifOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = ifOp.getLoc();
    assert(ifOp->getResults().empty() &&
           "Yul structured control flow does not support results yet");

    auto yulIfOp = r.create<yul::IfOp>(loc, TypeRange{}, adaptor.getCond());
    r.inlineRegionBefore(ifOp.getThenRegion(), yulIfOp.getThenRegion(),
                         yulIfOp.getThenRegion().end());
    lowerSolRegionTerminatorsToYul(yulIfOp.getThenRegion(), r);
    r.inlineRegionBefore(ifOp.getElseRegion(), yulIfOp.getElseRegion(),
                         yulIfOp.getElseRegion().end());
    lowerSolRegionTerminatorsToYul(yulIfOp.getElseRegion(), r);

    r.eraseOp(ifOp);
    return success();
  }
};

struct SwitchOpLowering : public OpConversionPattern<sol::SwitchOp> {
  using OpConversionPattern<sol::SwitchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::SwitchOp switchOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto yulSwitchOp = r.create<yul::SwitchOp>(
        switchOp.getLoc(), TypeRange{}, adaptor.getArg(), switchOp.getCases(),
        switchOp.getCaseRegions().size());

    r.inlineRegionBefore(switchOp.getDefaultRegion(),
                         yulSwitchOp.getDefaultRegion(),
                         yulSwitchOp.getDefaultRegion().end());
    lowerSolRegionTerminatorsToYul(yulSwitchOp.getDefaultRegion(), r);

    for (auto [srcRegion, dstRegion] :
         llvm::zip(switchOp.getCaseRegions(), yulSwitchOp.getCaseRegions())) {
      r.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
      lowerSolRegionTerminatorsToYul(dstRegion, r);
    }

    r.eraseOp(switchOp);
    return success();
  }
};

// (Copied and modified from clangir).
struct LoopOpInterfaceLowering
    : public OpInterfaceConversionPattern<sol::LoopOpInterface> {
  using OpInterfaceConversionPattern<
      sol::LoopOpInterface>::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(sol::LoopOpInterface op, ArrayRef<Value>,
                                ConversionPatternRewriter &r) const override {
    Operation *rawOp = op.getOperation();
    if (auto forOp = dyn_cast<sol::ForOp>(rawOp)) {
      auto yulForOp =
          r.create<yul::ForOp>(forOp.getLoc(), TypeRange{}, ValueRange{});
      r.inlineRegionBefore(forOp.getCond(), yulForOp.getCond(),
                           yulForOp.getCond().end());
      lowerSolRegionTerminatorsToYul(yulForOp.getCond(), r);
      r.inlineRegionBefore(forOp.getBody(), yulForOp.getBody(),
                           yulForOp.getBody().end());
      lowerSolRegionTerminatorsToYul(yulForOp.getBody(), r);
      r.inlineRegionBefore(forOp.getStep(), yulForOp.getStep(),
                           yulForOp.getStep().end());
      lowerSolRegionTerminatorsToYul(yulForOp.getStep(), r);
    } else if (auto whileOp = dyn_cast<sol::WhileOp>(rawOp)) {
      auto yulWhileOp =
          r.create<yul::WhileOp>(whileOp.getLoc(), TypeRange{}, ValueRange{});
      r.inlineRegionBefore(whileOp.getCond(), yulWhileOp.getCond(),
                           yulWhileOp.getCond().end());
      lowerSolRegionTerminatorsToYul(yulWhileOp.getCond(), r);
      r.inlineRegionBefore(whileOp.getBody(), yulWhileOp.getBody(),
                           yulWhileOp.getBody().end());
      lowerSolRegionTerminatorsToYul(yulWhileOp.getBody(), r);
    } else if (auto doWhileOp = dyn_cast<sol::DoWhileOp>(rawOp)) {
      auto yulDoWhileOp = r.create<yul::DoWhileOp>(doWhileOp.getLoc());
      r.inlineRegionBefore(doWhileOp.getBody(), yulDoWhileOp.getBody(),
                           yulDoWhileOp.getBody().end());
      lowerSolRegionTerminatorsToYul(yulDoWhileOp.getBody(), r);
      r.inlineRegionBefore(doWhileOp.getCond(), yulDoWhileOp.getCond(),
                           yulDoWhileOp.getCond().end());
      lowerSolRegionTerminatorsToYul(yulDoWhileOp.getCond(), r);
    } else {
      llvm_unreachable("unexpected Sol loop op");
    }

    r.eraseOp(rawOp);
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
    r.replaceOpWithNewOp<yul::FuncCallOp>(
        op, convertedResTys, op.getCalleeAttr(), adaptor.getOperands());
    return success();
  }
};

struct ReturnOpLowering
    : public CleanedOperandsLowering<sol::ReturnOp, ReturnOpLowering> {
  using Base = CleanedOperandsLowering<sol::ReturnOp, ReturnOpLowering>;
  using Base::Base;

  LogicalResult rewriteCleaned(sol::ReturnOp op, ConversionPatternRewriter &r,
                               ArrayRef<Value> operands) const {
    r.replaceOpWithNewOp<yul::FuncReturnOp>(op, operands);
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

    auto convertedFuncTy = cast<FunctionType>(
        getTypeConverter()->convertType(op.getFunctionType()));
    // FIXME: The location of the block arguments are lost here!
    auto newOp =
        r.create<yul::FuncOp>(op.getLoc(), op.getName(), convertedFuncTy);
    for (NamedAttribute attr : attrs)
      newOp->setAttr(attr.getName(), attr.getValue());
    r.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.getBody().end());
    r.eraseOp(op);
    return success();
  }
};

struct ICallOpLowering : public OpConversionPattern<sol::ICallOp> {
  // This lowering runs after Sol signatures are legalized but before sol.func
  // is lowered, so the dispatch table can still inspect the Sol symbol table.
  using OpConversionPattern<sol::ICallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sol::ICallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    Location loc = op.getLoc();
    evm::Builder evmB(getModule(op), r, loc);

    auto calleeArgs = adaptor.getOperands().drop_front();
    SmallVector<Type> convertedResTys;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                convertedResTys)))
      return failure();

    auto calleeTy = mlir::FunctionType::get(
        r.getContext(), calleeArgs.getTypes(), convertedResTys);

    // Collect functions with matching signature.
    Operation *symTab = SymbolTable::getNearestSymbolTable(op);
    bool callerRuntime = op->getParentOfType<sol::FuncOp>().getRuntime();
    SmallVector<int64_t> caseIds;
    SmallVector<sol::FuncOp> caseFns;
    symTab->walk([&](sol::FuncOp fn) {
      // At this stage the nearest Sol symbol table can still contain both
      // creation and runtime functions. Internal function pointers are only
      // valid within the caller's phase, so do not dispatch across that
      // boundary just because the id and signature match.
      if (fn.getRuntime() != callerRuntime)
        return;
      if (fn.getId() && fn.getFunctionType() == calleeTy) {
        caseFns.push_back(fn);
        caseIds.push_back(*fn.getId());
      }
    });

    auto calleeIntTy = cast<IntegerType>(adaptor.getCallee().getType());
    SmallVector<APInt> caseVals;
    caseVals.reserve(caseIds.size());
    for (int64_t caseId : caseIds)
      caseVals.emplace_back(calleeIntTy.getWidth(), caseId);
    auto caseIdsAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(caseVals.size()),
                              calleeIntTy),
        caseVals);
    auto switchOp = r.create<yul::SwitchOp>(
        loc, convertedResTys, adaptor.getCallee(), caseIdsAttr, caseIds.size());
    for (size_t i = 0; i < caseFns.size(); ++i) {
      r.setInsertionPointToStart(&switchOp.getCaseRegions()[i].emplaceBlock());
      auto call = r.create<yul::FuncCallOp>(
          loc, convertedResTys, FlatSymbolRefAttr::get(caseFns[i]), calleeArgs);
      r.create<yul::YieldOp>(loc, call.getResults());
    }

    r.setInsertionPointToStart(&switchOp.getDefaultRegion().emplaceBlock());
    evmB.genPanic(mlir::evm::PanicCode::InvalidInternalFunction);
    SmallVector<Value> undefs;
    undefs.reserve(op.getNumResults());
    for (Type ty : convertedResTys)
      undefs.push_back(r.create<LLVM::UndefOp>(loc, ty));
    r.create<yul::YieldOp>(loc, undefs);

    r.replaceOp(op, switchOp.getResults());
    return success();
  }
};

struct ContractOpLowering : public OpRewritePattern<sol::ContractOp> {
  const char *libAddrName = "library_deploy_address";

  using OpRewritePattern<sol::ContractOp>::OpRewritePattern;

  /// Generate the call value check.
  void genCallValChk(ModuleOp mod, PatternRewriter &r, Location loc) const {
    mlir::solgen::BuilderExt bExt(r, loc);
    evm::Builder evmB(mod, r, loc);

    auto callVal = r.create<yul::CallValOp>(loc);
    auto callValChk =
        bExt.genCmp(yul::CmpPredicate::ne, callVal, bExt.genI256Const(0));
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
    evm::Builder evmB(getModule(contrOp), r, loc);

    Value notDelegateCallCond;
    if (contrOp.getKind() == sol::ContractKind::Library) {
      auto libAddr = r.create<yul::LoadImmutableOp>(loc, r.getIntegerType(256),
                                                    libAddrName);
      auto currAddr = r.create<yul::AddressOp>(loc);
      notDelegateCallCond =
          bExt.genCmp(yul::CmpPredicate::eq, libAddr, currAddr);
    }

    // Do nothing if there are no interface functions.
    if (ifcFns.empty())
      return;

    // Generate `if iszero(lt(calldatasize(), 4))` and set the insertion point
    // to its then block.
    auto callDataSz = r.create<yul::CallDataSizeOp>(loc);
    auto callDataSzCmp =
        bExt.genCmp(yul::CmpPredicate::uge, callDataSz, bExt.genI256Const(4));
    auto ifOp = bExt.createIf(callDataSzCmp);
    OpBuilder::InsertionGuard insertGuard(r);
    r.setInsertionPointToStart(&ifOp.getThenRegion().front());

    // Load the selector from the calldata.
    auto callDataLd = r.create<yul::CallDataLoadOp>(loc, bExt.genI256Const(0));
    Value callDataSelector =
        r.create<yul::ArithShrOp>(loc, callDataLd, bExt.genI256Const(224));

    SmallVector<APInt, 4> selectorValues;
    selectorValues.reserve(selectors.size());
    for (uint32_t selector : selectors)
      selectorValues.emplace_back(/*numBits=*/256, selector);
    auto selectorsAttr = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get(static_cast<int64_t>(selectors.size()),
                                    r.getIntegerType(256)),
        selectorValues);

    // Generate the switch op.
    auto switchOp = r.create<yul::SwitchOp>(loc, /*resultTypes=*/TypeRange{},
                                            callDataSelector, selectorsAttr,
                                            selectors.size());

    // Generate the default block.
    {
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(r.createBlock(&switchOp.getDefaultRegion()));
      r.create<yul::YieldOp>(loc);
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
        genCallValChk(getModule(contrOp), r, loc);
      }

      // Decode the input parameters (if required).
      FunctionType origIfcFnTy = *ifcFnOp.getOrigFnType();
      SmallVector<Type> abiInputTys =
          getABITypes(r, origIfcFnTy.getInputs(), origIfcFnTy.getInputs(),
                      contrOp.getKind() == sol::ContractKind::Library);
      SmallVector<Type> abiResultTys =
          getABITypes(r, origIfcFnTy.getResults(), origIfcFnTy.getResults(),
                      contrOp.getKind() == sol::ContractKind::Library);
      std::vector<Value> decodedArgs;
      if (!abiInputTys.empty())
        evmB.genABITupleDecoding(abiInputTys,
                                 /*tupleStart=*/bExt.genI256Const(4),
                                 /*tupleEnd=*/callDataSz, decodedArgs,
                                 /*fromMem=*/false);

      // Generate the actual call.
      auto callOp = r.create<sol::CallOp>(loc, ifcFnOp, decodedArgs);

      // Encode the result using the ABI's tuple encoder.
      auto tupleStart = evmB.genFreePtr();
      mlir::Value tupleSize;
      if (!callOp.getResultTypes().empty()) {
        auto tupleEnd =
            evmB.genABIEncoding(abiResultTys, callOp.getResults(), tupleStart);
        tupleSize = r.create<yul::SubOp>(loc, tupleEnd, tupleStart);
      } else {
        tupleSize = bExt.genI256Const(0);
      }

      // Generate the return.
      assert(tupleSize);
      r.create<yul::ReturnOp>(loc, tupleStart, tupleSize);

      r.create<yul::YieldOp>(loc);
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
    ModuleOp mod = getModule(op);
    evm::Builder evmB(mod, r, loc);

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
      genCallValChk(mod, r, loc);
    } else {
      assert(ctor.getStateMutability());
      if (*ctor.getStateMutability() != sol::StateMutability::Payable)
        genCallValChk(mod, r, loc);
    }

    // Generate the call to constructor (if required).
    if (ctor && op.getKind() != sol::ContractKind::Library) {
      auto progSize = r.create<yul::DataSizeOp>(loc, creationObj.getName());
      auto codeSize = r.create<yul::CodeSizeOp>(loc);
      auto argSize = r.create<yul::SubOp>(loc, codeSize, progSize);
      Value tupleStart = evmB.genMemAlloc(argSize);
      r.create<yul::CodeCopyOp>(loc, tupleStart, progSize, argSize);
      std::vector<Value> decodedArgs;
      FunctionType ctorFnTy = *ctor.getOrigFnType();
      if (!ctorFnTy.getInputs().empty()) {
        evmB.genABITupleDecoding(
            ctorFnTy.getInputs(), tupleStart,
            /*tupleEnd=*/r.create<yul::AddOp>(loc, tupleStart, argSize),
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
      auto callDataSzIsZero =
          bExt.genCmp(yul::CmpPredicate::eq, callDataSz, bExt.genI256Const(0));
      auto ifOp = bExt.createIf(callDataSzIsZero);
      OpBuilder::InsertionGuard insertGuard(r);
      r.setInsertionPointToStart(&ifOp.getThenRegion().front());
      r.create<sol::CallOp>(loc, receiveFn, /*operands=*/ValueRange{});
      r.create<yul::StopOp>(loc);
    }

    // Generate fallback function.
    if (fallbackFn) {
      // Use the pre-lowering Sol signature: the live function type may already
      // be lowered here, and we need the Sol `bytes` types below.
      FunctionType fallbackFnTy = *fallbackFn.getOrigFnType();
      // Solidity allows either `fallback() external [payable]` (no params, no
      // returns) or `fallback(bytes calldata) external returns (bytes memory)`
      // (one of each); the two always agree.
      assert(fallbackFnTy.getNumInputs() == fallbackFnTy.getNumResults() &&
             "fallback has mismatched parameter/return counts");

      if (fallbackFn.getStateMutability() != sol::StateMutability::Payable) {
        genCallValChk(mod, r, loc);
      }

      // The `bytes calldata` parameter, when present, is the full call payload:
      // a `{offset, length}` reference of `{0, calldatasize()}`.
      SmallVector<Value, 1> fallbackArgs;
      if (fallbackFnTy.getNumInputs() == 1) {
        Value callDataSz = r.create<yul::CallDataSizeOp>(loc);
        fallbackArgs.push_back(
            bExt.genLLVMStruct({bExt.genI256Const(0), callDataSz}, loc));
      }

      auto call = r.create<sol::CallOp>(loc, fallbackFn, fallbackArgs);

      if (fallbackFnTy.getNumResults() == 1) {
        // The returned `bytes memory` is returned verbatim (not ABI-encoded),
        // i.e. `return(add(ret, 0x20), mload(ret))`.
        Value ret = call.getResult(0);
        Type retTy = fallbackFnTy.getResult(0);
        Value dataPtr = evmB.genDataAddrPtr(ret, sol::DataLocation::Memory, loc);
        Value len = evmB.genDynSize(ret, retTy, loc);
        r.create<yul::ReturnOp>(loc, dataPtr, len);
      } else {
        r.create<yul::StopOp>(loc);
      }

    } else {
      // TODO: Generate error message.
      evmB.genRevert(loc);
    }

    // StringLitOpLowering puts __data_in_code_* globals at module scope with
    // AddressOfOps inside yul::ObjectOps. For CODECOPY to bind to the right
    // object's data section, each global must live inside the ObjectOp that
    // references it
    llvm::DenseSet<LLVM::GlobalOp> toErase;
    mod->walk([&](LLVM::AddressOfOp addrOf) {
      StringRef name = addrOf.getGlobalName();
      if (!name.starts_with("__data_in_code_"))
        return;
      auto obj = addrOf->getParentOfType<yul::ObjectOp>();
      assert(obj);
      if (SymbolTable::lookupSymbolIn(obj, name))
        return;
      auto gOp = cast<LLVM::GlobalOp>(SymbolTable::lookupSymbolIn(mod, name));
      obj.getEntryBlock()->push_front(gOp->clone());
      toErase.insert(gOp);
    });
    for (LLVM::GlobalOp gOp : toErase)
      r.eraseOp(gOp);

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
           DynBytesToFixedBytesOpLowering, ExtFuncConstantOpLowering,
           ExtFuncSelectorOpLowering, ExtFuncAddrOpLowering,
           ArithBinOpLowering<sol::AddOp, yul::AddOp>,
           ArithBinOpLowering<sol::SubOp, yul::SubOp>,
           ArithBinOpLowering<sol::MulOp, yul::MulOp>,
           ArithBinOpLowering<sol::AndOp, yul::AndOp>,
           ArithBinOpLowering<sol::OrOp, yul::OrOp>,
           ArithBinOpLowering<sol::XorOp, yul::XOrOp>, NotOpLowering,
           DivOrModOpLowering<sol::DivOp, yul::ArithSDivOp, yul::ArithDivOp>,
           DivOrModOpLowering<sol::ModOp, yul::ArithSModOp, yul::ArithModOp>,
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
           GetCallDataOpLowering, GetReturnDataOpLowering,
           DefaultCallDataOpLowering, PushOpLowering,
           PopOpLowering, GepOpLowering, MapOpLowering, LoadOpLowering,
           LoadImmutableMetadataConversion, StoreOpLowering,
           DataLocCastOpLowering, LengthOpLowering, SliceOpLowering,
           CopyOpLowering, DeleteOpLowering, PushStringOpLowering,
           StringLitOpLowering, ConcatOpLowering>(tyConv, pats.getContext());
  pats.add<AddrOfOpLowering>(pats.getContext());
}

void evm::populateControlFlowPats(RewritePatternSet &pats,
                                  TypeConverter &tyConv) {
  pats.add<IfOpLowering, SwitchOpLowering, LoopOpInterfaceLowering>(
      tyConv, pats.getContext());
}

void evm::populateFuncBoundaryPats(RewritePatternSet &pats,
                                   TypeConverter &tyConv) {
  pats.add<CallOpLowering, ReturnOpLowering>(tyConv, pats.getContext());
}

void evm::populateFuncOpPats(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<FuncOpLowering>(tyConv, pats.getContext());
}

void evm::populateLateSolToYulPats(RewritePatternSet &pats,
                                   TypeConverter &tyConv) {
  pats.add<ICallOpLowering, LoadImmutableToYulLowering>(tyConv,
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
  pats.add<BlockHashOpLowering, BlobHashOpLowering, SelfdestructOpLowering>(
      tyConv, pats.getContext());
}

void evm::populateAbiPats(mlir::RewritePatternSet &pats,
                          mlir::TypeConverter &tyConv) {
  pats.add<EncodeOpLowering, DecodeOpLowering>(tyConv, pats.getContext());
}

void evm::populateExtCallPat(RewritePatternSet &pats, TypeConverter &tyConv) {
  pats.add<ExtCallOpLowering, ExtICallOpLowering, BareCallOpLowering,
           BareDelegateCallOpLowering, BareStaticCallOpLowering, TryOpLowering,
           NewOpLowering, CodeOpLowering, ObjectCodeOpLowering, SendOpLowering,
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
  populateControlFlowPats(pats, tyConv);
}

void evm::populateStage2Pats(RewritePatternSet &pats, TypeConverter &tyConv) {
  populateContractPat(pats);
  populateLateSolToYulPats(pats, tyConv);
}

//===- SolModifierOpLowering.cpp - Lower modifier ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that lowers ops related to modifiers (sol.modifier, sol.placeholder
// etc.)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Sol/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace mlir {
namespace sol {
#define GEN_PASS_DEF_MODIFIEROPLOWERINGPASS
#include "mlir/Dialect/Sol/Transforms/Passes.h.inc"
} // namespace sol
} // namespace mlir

using namespace mlir;

static constexpr const char *kBeforePlaceholderAttr =
    "sol.modifier.beforePlaceholder";

struct ModifierOpLoweringPass
    : public sol::impl::ModifierOpLoweringPassBase<ModifierOpLoweringPass> {

  // TODO: Move this to general utils.
  StringAttr getNearestUnusedSymFrom(Operation *op, StringAttr sym) {
    if (!SymbolTable::lookupNearestSymbolFrom(op, sym))
      return sym;

    unsigned i = 0;
    StringAttr newSym;
    do {
      newSym = StringAttr::get(op->getContext(),
                               sym.getValue() + std::to_string(i++));
    } while (SymbolTable::lookupNearestSymbolFrom(op, newSym));
    return newSym;
  }

  //
  // Terminologies, etc.
  //
  // - Modifier: The modifier.
  // - Function-like modifier: A modifier that's converted to a function.
  // - Caller function: The function that calls the modifier.
  // - Modifier function: The function that contains the body of the modifier
  // (in the context of a sol.modifier_call_blk).
  //
  // The caller function and the modifier function has the same signature. We
  // generate a modifier function for each sol.modifier_call_blk even if the
  // same modifier is called (since the call args can be different). The
  // modifier and the function-like modifier has the same signature.

  /// Returns the function-like modifier (or null) that the
  /// sol.modifier_call_blk calls.
  sol::FuncOp getFnLikeModifier(sol::ModifierCallBlkOp modifierCallBlk) {
    auto modifierCall = cast<sol::CallOp>(modifierCallBlk.getBody()->back());
    return dyn_cast<sol::FuncOp>(SymbolTable::lookupNearestSymbolFrom(
        modifierCall, modifierCall.getCalleeAttr()));
  }

  /// Creates and returns a function that contains the body of the modifier in
  /// the context of the sol.modifier_call_blk.
  sol::FuncOp createModifierFn(sol::ModifierCallBlkOp modifierCallBlk,
                               OpBuilder b) {
    OpBuilder::InsertionGuard guard(b);

    auto modifierCall = cast<sol::CallOp>(modifierCallBlk.getBody()->back());
    auto modifier = cast<sol::ModifierOp>(SymbolTable::lookupNearestSymbolFrom(
        modifierCall, modifierCall.getCalleeAttr()));
    auto callerFn = cast<sol::FuncOp>(modifierCallBlk->getParentOp());

    // Create the modifier function with caller function signature.
    StringRef modifierFnName =
        getNearestUnusedSymFrom(modifier, modifier.getNameAttr()).getValue();
    b.setInsertionPoint(modifier);
    auto modifierFn = b.create<sol::FuncOp>(modifier.getLoc(), modifierFnName,
                                            callerFn.getFunctionType());

    // Move the the sol.modifier_call_blk into the modifier function.
    modifierFn.getBody().takeBody(modifierCallBlk.getRegion());
    auto &modifierFnEntry = modifierFn.getBody().front();
    // sol.modifier_call_blk's block args should conform to the signature of the
    // modifier function.
    assert(modifierFnEntry.getArgumentTypes() ==
           modifierFn.getFunctionType().getInputs());

    // Clone the modifier's body and append it to the modifier function. The
    // cloned modifier's block args will be remapped to the modifier call args.
    auto &modifierEntry = modifier.getBody().front();
    IRMapping irMap;
    for (auto arg : modifierEntry.getArguments())
      irMap.map(arg, modifierCall.getArgOperands()[arg.getArgNumber()]);
    modifier.getBody().cloneInto(&modifierFn.getBody(), irMap);

    // "Replace" the modifier call with the entry block of the modifier.
    modifierCall.erase();
    assert(modifierFnEntry.getNextNode());
    modifierFnEntry.getOperations().splice(
        modifierFnEntry.end(), modifierFnEntry.getNextNode()->getOperations());
    modifierFnEntry.getNextNode()->erase();

    return modifierFn;
  }

  /// Replaces the sol.placeholders in the modifier function with a call to
  /// callee. This generates the return arg handling.  The callee is expected to
  /// have the same type of the modifier function as we forward args in the
  /// modifier function to the callee.
  void replacePlaceholders(sol::FuncOp modifierFn, sol::FuncOp callee,
                           OpBuilder b) {
    OpBuilder::InsertionGuard guard(b);

    FunctionType modifierFnTy = modifierFn.getFunctionType();
    assert(modifierFnTy == callee.getFunctionType());
    assert(modifierFnTy.getNumResults() <= 1 && "NYI");

    // Generate the alloca for the return arg.
    sol::AllocaOp retAddr;
    if (modifierFnTy.getNumResults() == 1) {
      b.setInsertionPointToStart(&modifierFn.getBlocks().front());
      retAddr = b.create<sol::AllocaOp>(
          modifierFn.getLoc(),
          sol::PointerType::get(b.getContext(), modifierFnTy.getResult(0),
                                sol::DataLocation::Stack));
    }

    // Replace sol.placeholders with a call to the callee.
    modifierFn.walk([&](sol::PlaceholderOp placeholder) {
      b.setInsertionPoint(placeholder);
      auto call = b.create<sol::CallOp>(placeholder.getLoc(), callee,
                                        modifierFn.getArguments());
      if (modifierFnTy.getNumResults() == 1)
        b.create<sol::StoreOp>(placeholder.getLoc(), call.getResult(0),
                               retAddr);
      placeholder.erase();
    });

    // Add the return arg in the sol.returns.
    if (modifierFnTy.getNumResults() == 1) {
      modifierFn.walk([&](sol::ReturnOp ret) {
        b.setInsertionPoint(ret);
        auto retVal = b.create<sol::LoadOp>(ret.getLoc(), retAddr);
        b.create<sol::ReturnOp>(ret.getLoc(), retVal.getResult());
        ret.erase();
      });
    }
  }

  /// Tries to convert the modifier to a function and returns it if it does. The
  /// transformed function will have the same signature of the modifier but no
  /// placeholders.
  ///
  /// We can do this iff:
  /// - All placeholder are immeditately before the return, then the placeholder
  /// acts like a return. In this case, the callee can call this as a regular
  /// function before its placeholder.
  ///
  /// - (TODO) Only one placeholder and it is at the beginning of the entry
  /// block, then the callee can call this as a regular function after its
  /// placeholder. TODO: 2 problems: (1) skipping stack allocas (impl mem2reg?)
  /// (2) If we append multiple calls at the callee's placeholder, the order
  /// gets reversed
  ///
  /// "Callee" here means the callee in the call-chain of modifiers that
  /// `lowerModifierCalls` generate.
  std::optional<sol::FuncOp> tryConvertingToFn(sol::ModifierOp modifier) {
    // Check if the placeholders act as return.
    auto walkRes = modifier.walk([&](sol::PlaceholderOp placeholder) {
      assert(placeholder->getNextNode());
      if (!isa<sol::ReturnOp>(placeholder->getNextNode()))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted())
      return std::nullopt;

    modifier.walk([&](sol::PlaceholderOp placeholder) { placeholder.erase(); });

    // Convert the modifier to a function.
    OpBuilder b(modifier.getContext());
    b.setInsertionPoint(modifier);
    auto replFn = b.create<sol::FuncOp>(modifier.getLoc(), modifier.getName(),
                                        modifier.getFunctionType());
    IRMapping mapper;
    modifier.getBody().cloneInto(&replFn.getBody(), mapper);
    replFn->setAttr(kBeforePlaceholderAttr, b.getUnitAttr());
    modifier.erase();
    return replFn;
  }

  /// Lowers all the modifier calls in the function.
  void lowerModifierCalls(sol::FuncOp callerFn) {
    if (callerFn.getBlocks().empty())
      return;

    OpBuilder b(callerFn.getContext());

    // Track all the sol.modifier_call_blk's.
    //
    // TODO: We could .reserve/bail out faster here if we track the modifier
    // count.
    SmallVector<sol::ModifierCallBlkOp> modifierCallBlks;
    Block &entryBlk = callerFn.getBlocks().front();
    for (Operation &op : entryBlk) {
      auto modifierCallBlk = dyn_cast<sol::ModifierCallBlkOp>(op);
      if (modifierCallBlk)
        modifierCallBlks.push_back(modifierCallBlk);
    }
    if (modifierCallBlks.empty())
      return;

    // Replace `callerFn` with a new function that calls the first modifier.
    sol::FuncOp newCallerFn = callerFn.cloneWithoutRegions();
    b.setInsertionPoint(callerFn);
    b.insert(newCallerFn);
    callerFn.setSymName(
        getNearestUnusedSymFrom(callerFn, callerFn.getSymNameAttr()));
    // Don't duplicate an interface function.
    callerFn.removeSelectorAttr();
    b.setInsertionPointToStart(newCallerFn.addEntryBlock());
    // The sol.placeholder will be replaced with the first modifier. This
    // simplifies the placeholder replacement loop.
    b.create<sol::PlaceholderOp>(callerFn.getLoc());
    b.create<sol::ReturnOp>(callerFn.getLoc());

    // Generate the chain of calls of modifiers and the caller function by
    // replacing the sol.placeholders.
    SmallVector<sol::FuncOp, 4> callChain{newCallerFn};
    for (sol::ModifierCallBlkOp modifierCallBlk : modifierCallBlks) {
      if (sol::FuncOp fnLikeModifer = getFnLikeModifier(modifierCallBlk)) {
        assert(fnLikeModifer->getAttr(kBeforePlaceholderAttr) && "NYI");

        // For function-like modifier, we generate the call to them before/after
        // (depending on the before/after-placeholder attribute) the
        // placeholders of the current tail of the `callChain`. They're not part
        // of the `callChain` as they won't have placeholders.

        // Clone the sol.modifier_call_blk ops before all the placeholders of
        // the `callChain`'s tail.
        IRMapping irMap;
        Block &callChainTailEntry = callChain.back().getBody().front();
        assert(modifierCallBlk.getBody()->getArgumentTypes() ==
               callChainTailEntry.getArgumentTypes());
        for (auto arg : modifierCallBlk.getBody()->getArguments())
          irMap.map(arg, callChainTailEntry.getArgument(arg.getArgNumber()));
        callChain.back().walk([&](sol::PlaceholderOp placeholder) {
          auto clonedModifierBlk =
              cast<sol::ModifierCallBlkOp>(modifierCallBlk->clone(irMap));
          placeholder->getBlock()->getOperations().splice(
              placeholder->getIterator(),
              clonedModifierBlk.getBody()->getOperations());
        });
      } else {
        callChain.push_back(createModifierFn(modifierCallBlk, b));
      }

      modifierCallBlk.erase();
    }
    callChain.push_back(callerFn);
    for (auto *it = callChain.begin(); it != callChain.end(); ++it) {
      if (it + 1 != callChain.end())
        replacePlaceholders(/*modifierFn=*/*it, /*callee=*/*(it + 1), b);
    }
  }

  void runOnOperation() override {
    getOperation().walk(
        [&](sol::ModifierOp modifier) { tryConvertingToFn(modifier); });
    getOperation().walk([&](sol::FuncOp fn) { lowerModifierCalls(fn); });
    getOperation().walk([&](sol::ModifierOp modifier) { modifier.erase(); });
  }
};

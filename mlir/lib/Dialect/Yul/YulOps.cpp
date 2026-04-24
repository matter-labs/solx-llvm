//===- YulOps.cpp - MLIR operations for Yul implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Yul operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::yul;

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << getValueAttr().getValue() << '_' << getType();
  setNameFn(getResult(), specialName.str());
}

LogicalResult FuncOp::verify() {
  auto i256Ty = IntegerType::get(getContext(), 256);
  auto fnTy = getFunctionType();

  for (Type inputTy : fnTy.getInputs())
    if (inputTy != i256Ty)
      return emitOpError("expects all input types to be i256");

  for (Type resultTy : fnTy.getResults())
    if (resultTy != i256Ty)
      return emitOpError("expects all result types to be i256");

  if (!getBody().empty()) {
    Block &entryBlock = getBody().front();
    if (entryBlock.getNumArguments() != fnTy.getNumInputs())
      return emitOpError("expects entry block argument count to match the "
                         "function type input count");

    for (BlockArgument arg : entryBlock.getArguments())
      if (arg.getType() != i256Ty)
        return emitOpError("expects all entry block argument types to be i256");

    for (Block &block : getBody())
      for (FuncReturnOp returnOp : block.getOps<FuncReturnOp>())
        if (returnOp->getNumOperands() != fnTy.getNumResults())
          return returnOp.emitOpError("expects operand count to match the "
                                      "parent yul.func result count");
  }

  return success();
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (!getElseRegion().empty())
    regions.push_back(RegionSuccessor(&getElseRegion()));
}

void SwitchOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  llvm::copy(getRegions(), std::back_inserter(regions));
}

void SwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = dyn_cast_or_null<IntegerAttr>(operands.front());
  if (!operandValue) {
    bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
    return;
  }

  unsigned liveIndex = getNumRegions() - 1;
  const auto it = llvm::find(getCases(), operandValue.getValue());
  if (it != getCases().end())
    liveIndex = std::distance(getCases().begin(), it);
  for (unsigned i = 0, e = getNumRegions(); i < e; ++i)
    bounds.emplace_back(/*lb=*/0, /*ub=*/i == liveIndex);
}

LogicalResult SwitchOp::verify() {
  auto i256Ty = IntegerType::get(getContext(), 256);

  if (getArg().getType() != i256Ty)
    return emitOpError("expects switch argument type to be i256");

  auto caseElementTy = getCases().getType().getElementType();
  if (caseElementTy != i256Ty)
    return emitOpError("expects case value type to be i256");

  if (getCases().getNumElements() !=
      static_cast<int64_t>(getCaseRegions().size()))
    return emitOpError("expects case value count to match case region count");

  return success();
}

void LoopOpInterface::getLoopOpSuccessorRegions(
    LoopOpInterface op, RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  assert(point.isParent() || point.getRegionOrNull());

  // Branching to first region: go to condition.
  if (point.isParent()) {
    regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
  }
  // Branching from condition: go to body or exit.
  else if (&op.getCond() == point.getRegionOrNull()) {
    regions.emplace_back(RegionSuccessor(op->getResults()));
    regions.emplace_back(&op.getBody(), op.getBody().getArguments());
  }
  // Branching from body: go to step or condition.
  else if (&op.getBody() == point.getRegionOrNull()) {
    auto *afterBody = (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
    regions.emplace_back(afterBody, afterBody->getArguments());
  }
  // Branching from step: go to condition.
  else if (op.maybeGetStep() == point.getRegionOrNull()) {
    regions.emplace_back(&op.getCond(), op.getCond().getArguments());
  } else {
    llvm_unreachable("unexpected branch origin");
  }
}

void ForOp::getSuccessorRegions(RegionBranchPoint point,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

SmallVector<Region *> ForOp::getLoopRegions() { return {&getBody()}; }

void SwitchOp::print(OpAsmPrinter &p) {
  p << ' ' << getArg() << " : " << getArg().getType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/getCasesAttrName().getValue());

  for (auto [val, region] : llvm::zip(getCases(), getCaseRegions())) {
    p.printNewline();
    p << "case " << val << ' ';
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }

  p.printNewline();
  p << "default ";
  p.printRegion(getDefaultRegion(), /*printEntryBlockArgs=*/false);
}

ParseResult SwitchOp::parse(OpAsmParser &p, OperationState &result) {
  OpAsmParser::UnresolvedOperand arg;
  IntegerType argTy;
  if (succeeded(p.parseOperand(arg))) {
    if (p.parseColon())
      return failure();
    if (p.parseType(argTy))
      return failure();
    if (p.resolveOperand(arg, argTy, result.operands))
      return failure();
  }

  SmallVector<APInt> caseVals;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    APInt value(argTy.getWidth(), 0);
    Region *region = result.addRegion();
    if (p.parseInteger(value) || p.parseRegion(*region))
      return failure();
    caseVals.push_back(value);
  }
  auto caseValsAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(caseVals.size())}, argTy),
      caseVals);
  result.addAttribute(getCasesAttrName(result.name), caseValsAttr);

  if (p.parseKeyword("default"))
    return failure();
  Region *defRegion = result.addRegion();
  if (p.parseRegion(*defRegion))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Yul/YulOps.cpp.inc"

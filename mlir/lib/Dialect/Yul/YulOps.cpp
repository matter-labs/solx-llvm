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

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::yul;

bool mlir::yul::isI256OrI256LLVMStruct(Type type) {
  if (type.isSignlessInteger(256))
    return true;

  auto structTy = dyn_cast<LLVM::LLVMStructType>(type);
  if (!structTy || structTy.isOpaque())
    return false;

  return llvm::all_of(structTy.getBody(), [](Type elementTy) {
    return elementTy.isSignlessInteger(256);
  });
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << getValueAttr().getValue() << '_' << getType();
  setNameFn(getResult(), specialName.str());
}

LogicalResult FuncOp::verify() {
  auto fnTy = getFunctionType();

  for (Type inputTy : fnTy.getInputs())
    if (!isI256OrI256LLVMStruct(inputTy))
      return emitOpError(
          "expects all input types to be i256 or LLVM structs containing only "
          "i256 elements");
  for (Type resultTy : fnTy.getResults())
    if (!isI256OrI256LLVMStruct(resultTy))
      return emitOpError(
          "expects all result types to be i256 or LLVM structs containing only "
          "i256 elements");

  if (!getBody().empty()) {
    Block &entryBlock = getBody().front();
    if (entryBlock.getNumArguments() != fnTy.getNumInputs())
      return emitOpError("expects entry block argument count to match the "
                         "function type input count");

    for (BlockArgument arg : entryBlock.getArguments())
      if (!isI256OrI256LLVMStruct(arg.getType()))
        return emitOpError("expects all entry block argument types to be i256 "
                           "or LLVM structs containing only i256 elements");

    for (auto [arg, inputTy] :
         llvm::zip(entryBlock.getArguments(), fnTy.getInputs()))
      if (arg.getType() != inputTy)
        return emitOpError("expects entry block argument types to match the "
                           "function type inputs");

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
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (!getElseRegion().empty())
    regions.push_back(RegionSuccessor(&getElseRegion()));
}

void SwitchOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
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
  auto caseElementTy = getCases().getType().getElementType();
  if (caseElementTy != getArg().getType())
    return emitOpError("expects case value type to match switch argument type");
  if (!caseElementTy.isInteger(256))
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

OperandRange ForOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getCond() &&
         "yul.for entry successor is expected to be the condition region");
  return getInitArgs();
}

MutableArrayRef<OpOperand> ForOp::getInitsMutable() {
  return (*this)->getOpOperands();
}

Block::BlockArgListType ForOp::getRegionIterArgs() {
  return getCond().front().getArguments();
}

std::optional<MutableArrayRef<OpOperand>> ForOp::getYieldedValuesMutable() {
  auto yieldOp = dyn_cast<YieldOp>(getStep().front().getTerminator());
  if (!yieldOp)
    return std::nullopt;
  return yieldOp.getOperandsMutable();
}

std::optional<ResultRange> ForOp::getLoopResults() { return getResults(); }

void WhileOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

SmallVector<Region *> WhileOp::getLoopRegions() { return {&getBody()}; }

OperandRange WhileOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getCond() &&
         "yul.while entry successor is expected to be the condition region");
  return getInitArgs();
}

MutableArrayRef<OpOperand> WhileOp::getInitsMutable() {
  return (*this)->getOpOperands();
}

Block::BlockArgListType WhileOp::getRegionIterArgs() {
  return getCond().front().getArguments();
}

std::optional<MutableArrayRef<OpOperand>> WhileOp::getYieldedValuesMutable() {
  auto yieldOp = dyn_cast<YieldOp>(getBody().front().getTerminator());
  if (!yieldOp)
    return std::nullopt;
  return yieldOp.getOperandsMutable();
}

std::optional<ResultRange> WhileOp::getLoopResults() { return getResults(); }

void DoWhileOp::getSuccessorRegions(RegionBranchPoint point,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  LoopOpInterface::getLoopOpSuccessorRegions(*this, point, regions);
}

SmallVector<Region *> DoWhileOp::getLoopRegions() { return {&getBody()}; }

OperandRange DoWhileOp::getEntrySuccessorOperands(RegionBranchPoint point) {
  assert(point == getBody() &&
         "yul.do entry successor is expected to be the body region");
  auto operandEnd = (*this)->operand_end();
  return OperandRange(operandEnd, operandEnd);
}

MutableArrayRef<OpOperand> DoWhileOp::getInitsMutable() { return {}; }

Block::BlockArgListType DoWhileOp::getRegionIterArgs() {
  return Block::BlockArgListType();
}

std::optional<MutableArrayRef<OpOperand>> DoWhileOp::getYieldedValuesMutable() {
  auto yieldOp = dyn_cast<YieldOp>(getBody().front().getTerminator());
  if (!yieldOp)
    return std::nullopt;
  return yieldOp.getOperandsMutable();
}

std::optional<ResultRange> DoWhileOp::getLoopResults() {
  return (*this)->getResults();
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << ' ' << getArg() << " : " << getArg().getType();
  if (!getResultTypes().empty())
    p << " -> " << getResultTypes();
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
  if (p.parseOperand(arg) || p.parseColon() || p.parseType(argTy) ||
      p.resolveOperand(arg, argTy, result.operands) ||
      p.parseOptionalArrowTypeList(result.types))
    return failure();

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

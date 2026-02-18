//===- SolOps.td - Solidity dialect operations -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_SOLOPS_TD
#define MLIR_DIALECT_SOL_SOLOPS_TD

include "SolBase.td"
include "SolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// TODO: Add summary and description to all the op definitions.
// TODO: Impl SymbolUserOpInterface's.

// TODO: Implement verifiers for all the ops.
//
// Most of the high level sol dialect ops (excluding sol.func + related ops, yul
// specific ops) should verify that their types (ssa argument types and
// TypeAttr) are high level ((un)signed int types, sol dialect types etc.).

// TODO: Is using builtin.unrealized_conversion_cast better+faster? Can we
// legalize it without a separate reconcile pass?
def Sol_ConvCastOp : Sol_Op<"conv_cast", [Pure]> {
  let summary = "No-op cast that casts to the remapped value in conversion";

  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `->` type($out)";

  let hasFolder = 1;
}

def Sol_ConstantOp : Op<Sol_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]
>{
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def Sol_CastOp : Sol_Op<"cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins Sol_Int:$inp);
  let results = (outs Sol_Int:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
  let hasFolder = 1;
}

// This operation can panic.
def Sol_EnumCastOp : Sol_Op<"enum_cast",
  [DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins Sol_Int:$inp);
  let results = (outs Sol_EnumType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

// TODO: folder
def Sol_BytesCastOp : Sol_Op<"bytes_cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins AnyTypeOf<[Sol_BytesType, Sol_Int]>:$inp);
  let results = (outs AnyTypeOf<[Sol_BytesType, Sol_Int]>:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

// TODO: Add folders for the c0 case.

// Checked arith ops are not speculatable, but unchecked are (including div and
// mod because of evm semantics).

def Sol_CAddOp : Sol_IntArithOp<"cadd", [Commutative]>;
def Sol_CSubOp : Sol_IntArithOp<"csub">;
def Sol_CMulOp : Sol_IntArithOp<"cmul", [Commutative]>;
def Sol_CDivOp : Sol_IntArithOp<"cdiv">;
def Sol_CExpOp : Sol_IntArithOp<"cexp">;

def Sol_AddOp : Sol_UncheckedIntArithOp<"add", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_SubOp : Sol_UncheckedIntArithOp<"sub"> {
  // TODO: folder, canonicalizer
}
def Sol_MulOp : Sol_UncheckedIntArithOp<"mul", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_DivOp : Sol_UncheckedIntArithOp<"div"> {
  // TODO: folder, canonicalizer
}
def Sol_ModOp : Sol_UncheckedIntArithOp<"mod"> {
  // TODO: folder, canonicalizer
}
def Sol_ExpOp : Sol_UncheckedIntArithOp<"exp"> {
  // TODO: folder, canonicalizer
}
def Sol_ShlOp : Sol_UncheckedIntArithOp<"shl"> {
  // TODO: folder, canonicalizer
}
def Sol_ShrOp : Sol_UncheckedIntArithOp<"shr"> {
  // TODO: folder, canonicalizer
}
def Sol_AndOp : Sol_UncheckedIntArithOp<"and", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_OrOp : Sol_UncheckedIntArithOp<"or", [Commutative]> {
  // TODO: folder, canonicalizer
}
def Sol_XorOp : Sol_UncheckedIntArithOp<"xor", [Commutative]> {
  // TODO: folder, canonicalizer
}

def Sol_AddModOp : Sol_ModOp<"addmod">;
def Sol_MulModOp : Sol_ModOp<"mulmod">;

def Sol_Keccak256Op : Sol_Op<"keccak256"> {
  let arguments = (ins Sol_StringType:$addr);
  let results = (outs Sol_BytesType:$result);
}

def Sol_Sha256Op : Sol_HashPreCompileOp<"sha256">;
def Sol_Ripemd160Op : Sol_HashPreCompileOp<"ripemd160">;

def Sol_EcrecoverOp : Sol_Op<"ecrecover"> {
  let arguments = (ins Sol_BytesType:$hash, AnyUnsignedInteger:$v,
      Sol_BytesType:$r, Sol_BytesType:$s);
  let results = (outs Sol_AddressType:$result);
}

def Sol_CmpOp : Sol_Op<"cmp", [Pure, SameTypeOperands]> {
  let arguments = (ins Sol_CmpPredicateAttr:$predicate, Sol_Int:$lhs,
                       Sol_Int:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
  // TODO: folder, canonicalizer
}

def Sol_AllocaOp : Sol_Op<"alloca"> {
  let arguments = (ins TypeAttr:$alloc_type);
  let results = (outs Res<Sol_PtrType, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let builders = [OpBuilder<(ins "Type":$allocType), [{
    $_state.addAttribute(getAllocTypeAttrName($_state.name),
                         mlir::TypeAttr::get(allocType));
    $_state.addTypes(allocType);
  }]>];
  let skipDefaultBuilders = 1;

  let hasCustomAssemblyFormat = 1;
}

def Sol_MallocOp : Sol_Op<"malloc"> {
  let arguments = (ins UnitAttr:$zero_init, Optional<Sol_Int>:$size);
  let results = (outs Sol_AggType:$addr);

  // FIXME: Generated builders should include a version that defaults $size to
  // {}

  // FIXME: Comma after `type($size)`
  let assemblyFormat = [{
    ($size^)? (`zero_init` $zero_init^)? attr-dict `:`
      type($size) type($addr)
  }];
}

def Sol_GetCallDataOp : Sol_Op<"get_calldata"> {
  let results = (outs Sol_AggType:$addr);
  let assemblyFormat = [{ attr-dict `:` type($addr) }];

  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(sol::StringType::get($_builder.getContext(),
                                            sol::DataLocation::CallData));
    }]>
  ];
  let skipDefaultBuilders = 1;
}

def Sol_PushOp : Sol_Op<"push"> {
  let arguments = (ins AnyType:$inp);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);

  let builders = [
    OpBuilder<(ins "Value":$inp)>
  ];

  let assemblyFormat = "$inp attr-dict `:` type($inp) `->` type($addr)";
}

def Sol_PopOp : Sol_Op<"pop"> {
  let arguments = (ins AnyType:$inp);
  let assemblyFormat = "$inp attr-dict `:` type($inp)";
}

def Sol_ArrayLitOp : Sol_Op<"array_lit"> {
  let arguments = (ins Variadic<AnyType>:$ins);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);
  let assemblyFormat = "$ins attr-dict `:` `(` type($ins) `)` `->` type($addr)";
}

def Sol_StateVarOp : Sol_Op<"state_var", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Sol_ImmutableOp : Sol_Op<"immutable", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Sol_AddrOfOp : Sol_Op<"addr_of", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$var);
  // FIXME: AnyType -> Sol_PtrTy causes the asm printer to miss the prefix!
  let results = (outs AnyType:$addr);

  let assemblyFormat = "$var attr-dict `:` type($addr)";
}

def Sol_GepOp : Sol_Op<"gep"> {
  let arguments = (ins AnyType:$baseAddr, Sol_Int:$idx);
  let results = (outs Sol_RefType:$addr);

  let builders = [
    OpBuilder<(ins "Value":$baseAddr, "Value":$idx)>
  ];

  // TODO: Verifier:
  // - Base address type is an array or a struct type.
  // - Element type should not be a pointer type.
  // - Index should be within bounds for static arrays and structs.
  // - Struct index should be an i64.

  let assemblyFormat = [{
    $baseAddr `,` $idx attr-dict `:` type($baseAddr) `,` type($idx)
      `,` type($addr)
  }];
}

def Sol_MapOp : Sol_Op<"map"> {
  let arguments = (ins AnyType:$mapping, AnyType:$key);
  let results = (outs AnyType:$addr);

  let assemblyFormat = [{
    $mapping `,` $key attr-dict `:` type($mapping) `,` type($key) `,`
                                    type($addr)
  }];
}

def Sol_CopyOp : Sol_Op<"copy"> {
  let arguments = (ins AnyType:$src, AnyType:$dst);

  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

// TODO: Simplify the asm format by inferring element type of Sol_RefType.

def Sol_LoadOp : Sol_Op<"load", [
  AlwaysSpeculatable,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
>{
  let arguments = (ins Sol_RefType:$addr);
  let results = (outs AnyType:$out);

  let builders = [
    OpBuilder<(ins "Value":$addr), [{
      build($_builder, $_state, getEltType(addr.getType()), addr);
    }]>
  ];

  let assemblyFormat = "$addr attr-dict `:` type($addr) `,` type($out)";
}

def Sol_LoadImmutableOp : Sol_Op<"load_immutable", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyType:$val);

  let assemblyFormat = "$name attr-dict `:` type($val)";
}

def Sol_LibAddrOp : Sol_Op<"lib_addr", [Pure]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Sol_AddressType:$val);

  let assemblyFormat = "$name attr-dict `:` type($val)";
}

def Sol_StoreOp : Sol_Op<"store", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
>{
  let arguments = (ins AnyType:$val, Sol_RefType:$addr);

  let assemblyFormat = [{
    $val `,` $addr attr-dict `:` type($val) `,` type($addr)
  }];
}

def Sol_DataLocCastOp : Sol_Op<"data_loc_cast"> {
  let arguments = (ins AnyType:$inp);
  let results = (outs AnyType:$out);

  let assemblyFormat = [{ $inp attr-dict `:` type($inp) `,` type($out) }];
}

def Sol_LengthOp : Sol_Op<"length"> {
  let arguments = (ins AnyType:$inp);
  let results = (outs UI256:$len);

  let assemblyFormat = [{ $inp attr-dict `:` type($inp) }];
}

def Sol_SliceOp : Sol_Op<"slice"> {
  let arguments = (ins AnyType:$arr, Sol_Int:$start, Sol_Int:$end);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    $arr `[` $start `:` $end `]` attr-dict `:` type($arr) `,` type($start) `,`
                                               type($end) `->` type($res)
  }];
}

def Sol_EncodeOp : Sol_Op<"encode"> {
  let arguments = (ins Variadic<AnyType>:$ins);
  // FIXME: AnyType -> Sol_StringType causes the asm printer to miss the prefix!
  let results = (outs AnyType:$res);
  let assemblyFormat = "($ins^ `:` type($ins))? `:` type($res) attr-dict";
}

def Sol_DecodeOp : Sol_Op<"decode"> {
  let arguments = (ins AnyType:$addr);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "$addr `:` type($addr) `->` type($outs) attr-dict";
}

def Sol_EmitOp : Sol_Op<"emit"> {
  let arguments = (ins Variadic<AnyType>:$args, I8Attr:$indexedArgsCount,
                       OptionalAttr<StrAttr>:$signature);

  let builders = [
    OpBuilder<(ins "std::vector<Value>":$indexedArgs,
                    CArg<"std::vector<Value>", "{}">:$nonIndexedArgs,
                    CArg<"std::optional<std::string>",
                         "std::nullopt">:$signature), [{
      $_state.addOperands(indexedArgs);
      $_state.addOperands(nonIndexedArgs);

      IntegerAttr indexedArgsCountAttr =
          $_builder.getIntegerAttr($_builder.getIntegerType(8),
                                   indexedArgs.size());
      $_state.addAttribute(getIndexedArgsCountAttrName($_state.name),
                           indexedArgsCountAttr);

      if (signature)
        $_state.addAttribute(getSignatureAttrName($_state.name),
                             $_builder.getStringAttr(*signature));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    OperandRange getIndexedArgs() {
      return OperandRange(getOperands().begin(), getOperands().begin() +
                          getIndexedArgsCount());
    }
    OperandRange getNonIndexedArgs() {
      return OperandRange(getOperands().begin() + getIndexedArgsCount(),
                          getOperands().end());
    }
  }];

  // TODO: Verifier: ($indexedArgsCount <= 3)

  let hasCustomAssemblyFormat = 1;
}

def Sol_RevertOp : Sol_Op<"revert"> {
  // $signature represents the error call's signature if $call is set.
  let arguments = (ins Variadic<AnyType>:$args, StrAttr:$signature,
                       UnitAttr:$call);

  let assemblyFormat = "$signature ($args^ `:` type($args))? attr-dict";
}

def Sol_AssertOp : Sol_Op<"assert"> {
  let arguments = (ins I1:$cond);
  let assemblyFormat = "$cond attr-dict";
}

def Sol_RequireOp : Sol_Op<"require"> {
  // $msg represents the error call's signature if $call is set.
  let arguments = (ins I1:$cond, StrAttr:$msg,
                       Variadic<AnyType>:$args, UnitAttr:$call);

  let assemblyFormat = [{
    $cond `,` $msg `(` ($args^ `:` type($args))? `)` attr-dict
  }];
}

// TODO: Implement verifiers for sol.break and sol.continue that checks if it
// has an ancestor loop op.

def Sol_BreakOp : Sol_Op<"break", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Sol_ContinueOp : Sol_Op<"continue", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Sol_ConditionOp : Sol_Op<"condition", [
  Terminator,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
                            ["getSuccessorRegions"]>]
> {
  let arguments = (ins I1:$condition);

  let assemblyFormat = "$condition attr-dict";
}

def Sol_YieldOp : Sol_Op<"yield", [ReturnLike, Terminator,
  ParentOneOf<["IfOp", "SwitchOp", "TryOp", "WhileOp", "DoWhileOp", "ForOp"]>]
> {
  let arguments = (ins Variadic<AnyType>:$ins);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "($ins^ `:` type($ins))? attr-dict";
}

def Sol_IfOp : Sol_Op<"if",
  [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
   AutomaticAllocationScope, NoRegionArguments]
> {
  let arguments = (ins I1:$cond);
  let regions = (region MinSizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let assemblyFormat = "$cond $thenRegion `else` $elseRegion attr-dict";
}

def Sol_SwitchOp : Sol_Op<"switch", [
  AutomaticAllocationScope, NoRegionArguments,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                            ["getRegionInvocationBounds"]>
]> {
  let arguments = (ins AnyInteger:$arg, AnyIntElementsAttr:$cases);
  let regions = (region MinSizedRegion<1>:$defaultRegion,
                        VariadicRegion<MinSizedRegion<1>>:$caseRegions);

  let hasCustomAssemblyFormat = 1;
}

class Sol_WhileOpBase<string mnemonic> : Sol_Op<mnemonic, [
  Sol_LoopOpInterface, NoRegionArguments, AutomaticAllocationScope
]>;

def Sol_WhileOp : Sol_WhileOpBase<"while"> {
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body);

  let assemblyFormat = "$cond `do` $body attr-dict";
}

def Sol_DoWhileOp : Sol_WhileOpBase<"do"> {
  let regions = (region MinSizedRegion<1>:$body, SizedRegion<1>:$cond);

  let assemblyFormat = "$body `while` $cond attr-dict";

  let extraClassDeclaration = [{
    Region &getEntry() { return getBody(); }
  }];
}

def Sol_ForOp : Sol_Op<"for", [Sol_LoopOpInterface, NoRegionArguments,
                               AutomaticAllocationScope]> {
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body,
                        SizedRegion<1>:$step);

  let assemblyFormat = [{
    `cond` $cond
    `body` $body
    `step` $step
    attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::Region *maybeGetStep() { return &getStep(); }
  }];
}

def Sol_TryOp : Sol_Op<"try",
  [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
   AutomaticAllocationScope]
> {
  let regions = (region AnyRegion:$successRegion, AnyRegion:$panicRegion,
                        AnyRegion:$errorRegion, AnyRegion:$fallbackRegion);
  let arguments = (ins I1:$status);

  let assemblyFormat = [{
    $status $successRegion `panic` $panicRegion `error` $errorRegion
      `fallback` $fallbackRegion attr-dict
  }];
}

// TODO: We need a trait like "HasAncestor<ModifierOp>".
def Sol_PlaceholderOp : Sol_Op<"placeholder"> {
  let assemblyFormat = "attr-dict";
}

def Sol_ModifierCallBlkOp : Sol_Op<"modifier_call_blk",
                                   [SingleBlock, NoTerminator,
                                    IsolatedFromAbove]> {
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins), [{ $_state.addRegion()->emplaceBlock(); }]>
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = "$bodyRegion attr-dict";
}

// TODO: Modifiers doesn't return anything and it can't be external. Should we
// replace the FunctionOpInterface here with something more appropriate?
def Sol_ModifierOp : Sol_Op<"modifier",
                            [AutomaticAllocationScope, FunctionOpInterface,
                             CallableOpInterface, IsolatedFromAbove, Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type), [{
      $_state.addRegion();
      $_state.addAttribute(getFunctionTypeAttrName($_state.name),
                           TypeAttr::get(type));
      $_state.addAttribute(SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return &getBody(); }

    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }

    ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//
// TODO: Move function ops to another dialect
//

def Sol_DefaultFuncConstantOp : Sol_Op<"default_func_constant", [Pure]> {
  let results = (outs AnyType:$addr);

  let builders = [
    OpBuilder<(ins), [{
      auto fnTy = $_builder.getFunctionType({}, {});
      $_state.addTypes(sol::FuncRefType::get($_builder.getContext(), fnTy));
    }]>
  ];

  let assemblyFormat = "attr-dict `:` type($addr)";
}

def Sol_FuncConstantOp : Sol_Op<"func_constant", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$sym);
  let results = (outs AnyType:$addr);

  let assemblyFormat = "$sym attr-dict `:` type($addr)";
}

def ICallOp : Sol_Op<"icall", [CallOpInterface]> {
  // FIXME: AnyType -> Sol_FuncRefType causes the asm printer to miss the prefix!
  let arguments = (ins AnyType:$callee,
                       Variadic<AnyType>:$callee_operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>);

  // TODO: Custom builder that doesn't ask for arg/res attrs.

  let extraClassDeclaration = [{
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, cast<Value>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:`
      type($callee) `,` functional-type($callee_operands, results)
  }];
}

def CallOp : Sol_Op<"call", [CallOpInterface]> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute(getCalleeAttrName($_state.name),
                           SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute(getCalleeAttrName($_state.name), callee);
      $_state.addTypes(resTys);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), resTys, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$resTys,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            resTys, operands);
    }]>];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Sol_ThisOp : Sol_Op<"this"> {
  let results = (outs Sol_AddressType:$addr);

  let assemblyFormat = "attr-dict `:` type($addr)";
}

def Sol_ExtCallOp : Sol_Op<"ext_call", [CallOpInterface]> {
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$ins,
                       Sol_AddressType:$addr, UI256:$gas, UI256:$val, UnitAttr:$try_call,
                       UnitAttr:$static_call, UnitAttr:$delegate_call,
                       I32Attr:$selector,
                       TypeAttrOf<FunctionType>:$callee_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs I1:$status, Variadic<AnyType>);

  // TODO: Custom builder that doesn't ask for arg/res attrs.

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      return {operand_begin() + 1, operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/1,
          /*length=*/getNumOperands() - 1);
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $ins `)` `at` $addr `gas` $gas `value` $val attr-dict `:`
      functional-type($ins, results)
  }];
}

def Sol_NewOp : Sol_Op<"new", [AttrSizedOperandSegments]> {
  let arguments = (ins StrAttr:$objName, UI256:$val,
                   Optional<UI256>:$salt,
                   Variadic<AnyType>:$ctorArgs);
  let results = (outs Sol_AddressType:$out);

  let assemblyFormat = [{
    $objName `value` `=` $val (`salt` `=` $salt^)?
    `ctor` `(` ($ctorArgs^ `:` type($ctorArgs))? `)` attr-dict `:` type($out)
  }];
}

def Sol_CodeOp : Sol_Op<"code"> {
  let arguments = (ins Sol_AddressType:$contAddr);
  // FIXME: AnyType -> Sol_StringType causes the asm printer to miss the prefix!
  let results = (outs AnyType:$out);

  let builders = [
    OpBuilder<(ins "Value":$contAddr)>
  ];

  let assemblyFormat = [{
    $contAddr attr-dict `:` type($contAddr) `->` type($out)
  }];
}

def Sol_CodeHashOp : Sol_Op<"code_hash", [Pure]> {
  let arguments = (ins Sol_AddressType:$contAddr);
  let results = (outs UI256:$out);

  let assemblyFormat = [{
    $contAddr attr-dict `:` type($contAddr) `->` type($out)
  }];
}

def Sol_ReturnOp : Sol_Op<"return", [Pure, Terminator]> {
  // TODO: Verifier should check if a func op is an ancestor. ParentOneOf
  // doesn't work for this.
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Sol_FuncOp : Sol_Op<"func", [AutomaticAllocationScope, CallableOpInterface,
                                 FunctionOpInterface, IsolatedFromAbove,
                                 Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<I32Attr>:$selector,
                       OptionalAttr<TypeAttrOf<FunctionType>>:$orig_fn_type,
                       UnitAttr:$runtime,
                       OptionalAttr<Sol_FunctionKindAttr>:$kind,
                       OptionalAttr<Sol_StateMutabilityAttr>:$state_mutability,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       // Unique id used for generating dispatch table in the
                       // indirect call lowering
                       OptionalAttr<I64Attr>:$id);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      "StateMutability":$stateMutability,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    // Copied from func.func.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ArrayAttr getCallableArgAttrs() { return getArgAttrs().value_or(nullptr); }

    ArrayAttr getCallableResAttrs() { return getResAttrs().value_or(nullptr); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Sol_ContractOp : Sol_Op<"contract", [Symbol, SymbolTable, SingleBlock,
                                         IsolatedFromAbove, NoTerminator]> {
  let arguments = (ins SymbolNameAttr:$sym_name, Sol_ContractKindAttr:$kind);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$sym_name $bodyRegion attr-dict";
}

#endif // MLIR_DIALECT_SOL_SOLOPS_TD

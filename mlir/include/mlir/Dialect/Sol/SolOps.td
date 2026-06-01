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
  let arguments = (ins AnyInteger:$inp);
  let results = (outs AnyInteger:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
  let hasFolder = 1;
}

def Sol_AddressCastOp : Sol_Op<"address_cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments =
      (ins AnyTypeOf<[AnyInteger, Sol_AddressType, Sol_ContractType,
                      Sol_FixedBytesType]>:$inp);
  let results = (outs AnyTypeOf<[AnyInteger, Sol_AddressType, Sol_ContractType,
                                 Sol_FixedBytesType]>:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
  let hasFolder = 1;
}

def Sol_ContractCastOp : Sol_Op<"contract_cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins Sol_ContractType:$inp);
  let results = (outs Sol_ContractType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
  let hasFolder = 1;
}

// This operation can panic.
def Sol_EnumCastOp : Sol_Op<"enum_cast",
  [DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins Sol_Int:$inp);
  let results = (outs Sol_Int:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

// TODO: folder
def Sol_BytesCastOp : Sol_Op<"bytes_cast",
  [Pure, DeclareOpInterfaceMethods<CastOpInterface>]
>{
  let arguments = (ins AnyTypeOf<[Sol_ByteType, Sol_FixedBytesType,
                                  Sol_Int]>:$inp);
  let results = (outs AnyTypeOf<[Sol_ByteType, Sol_FixedBytesType,
                                 Sol_Int]>:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

def Sol_DynBytesToFixedBytesOp : Sol_Op<"dyn_bytes_to_fixedbytes"> {
  let arguments = (ins Sol_StringType:$inp);
  let results = (outs Sol_FixedBytesType:$out);

  let assemblyFormat = "$inp attr-dict `:` type($inp) `to` type($out)";
}

// TODO: Add folders for the c0 case.

// Checked arith ops are not speculatable, but unchecked are (including div and
// mod because of evm semantics).

def Sol_CAddOp : Sol_IntArithOp<"cadd", [Commutative]>;
def Sol_CSubOp : Sol_IntArithOp<"csub">;
def Sol_CMulOp : Sol_IntArithOp<"cmul", [Commutative]>;
def Sol_CDivOp : Sol_IntArithOp<"cdiv">;
def Sol_CExpOp : Sol_IntExpOp<"cexp">;

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
def Sol_ExpOp : Sol_IntExpOp<"exp", [Pure]> {
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

def Sol_NotOp : Sol_Op<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "Represents Solidity bitwise not";

  let arguments = (ins Sol_Int:$value);
  let results = (outs Sol_Int:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Sol_AddModOp : Sol_ModOp<"addmod">;
def Sol_MulModOp : Sol_ModOp<"mulmod">;

def Sol_Keccak256Op : Sol_Op<"keccak256"> {
  let arguments = (ins Sol_StringType:$addr);
  let results = (outs Sol_FixedBytesType:$result);
}

def Sol_Sha256Op : Sol_HashPreCompileOp<"sha256">;
def Sol_Ripemd160Op : Sol_HashPreCompileOp<"ripemd160">;

def Sol_EcrecoverOp : Sol_Op<"ecrecover"> {
  let arguments = (ins Sol_FixedBytesType:$hash, AnyUnsignedInteger:$v,
      Sol_FixedBytesType:$r, Sol_FixedBytesType:$s);
  let results = (outs Sol_AddressType:$result);
}

def Sol_CmpOp : Sol_Op<"cmp", [Pure, SameTypeOperands]> {
  let arguments = (ins Sol_CmpPredicateAttr:$predicate,
      AnyTypeOf<[Sol_Int, Sol_AddressType, Sol_ContractType, Sol_FixedBytesType,
                 Sol_ExtFuncRefType, Sol_FuncRefType]>:$lhs,
      AnyTypeOf<[Sol_Int, Sol_AddressType, Sol_ContractType, Sol_FixedBytesType,
                 Sol_ExtFuncRefType, Sol_FuncRefType]>:$rhs);
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

def Sol_GetReturnDataOp : Sol_Op<"get_returndata"> {
  let summary = "Copies the current returndata (from `start`) into a fresh memory `bytes`.";
  let description = [{
    Yields the returndata of the most recent external call, starting at byte
    offset `$start`, as a memory-located `bytes` value (`returndatasize` -
    `start` bytes `returndatacopy`d into a freshly allocated dynamic array).
    `start` is `0` for the whole returndata (low-level `catch (bytes)`); typed
    `catch Error/Panic` pass `4` to skip the error selector before ABI-decoding.
  }];
  let arguments = (ins Sol_Int:$start);
  let results = (outs Sol_AggType:$addr);
  let assemblyFormat = [{ $start attr-dict `:` type($start) `->` type($addr) }];

  let builders = [
    OpBuilder<(ins "Value":$start), [{
      $_state.addOperands(start);
      $_state.addTypes(sol::StringType::get($_builder.getContext(),
                                            sol::DataLocation::Memory));
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

// String push operations are separate from Sol_PushOp because Solidity strings
// use a dual storage encoding that requires special handling:
//
//   - Short strings (< 32 bytes): data and length are packed into a single
//     storage slot ("in-place" encoding).
//   - Long strings (>= 32 bytes): the slot holds `length * 2 + 1` and the
//     byte data is stored starting at keccak256(slot) ("out-of-place"
//     encoding).
//
// Appending a byte may trigger a transition from in-place to out-of-place
// encoding at the 31-byte boundary, which generic Sol_PushOp does not handle.
//
// The two ops correspond to Solidity's two push() overloads on dynamic byte
// arrays:
//   - Sol_PushStringOp    : str.push(x) -- appends a specific byte value.
//   - Sol_PushOp          : str.push()  -- appends a zero byte and returns a
//     storage reference ({slot, offset}) to the new element so the caller can
//     write to it later.
// We have two separate operations because implementing push(x) as push() = x
// would introduce significant gas overhead. Storage would need to be read and
// written twice: first when updating the length of the packed string
// (inside push()), and again when inserting the new value.
//
def Sol_PushStringOp : Sol_Op<"push_string"> {
  let arguments = (ins Sol_StringType:$addr,
      AnyTypeOf<[Sol_FixedBytesType, Sol_Int]>:$value);
  let assemblyFormat =
      "$addr `,` $value attr-dict `:` type($addr) `,` type($value)";
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
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type,
                       I256Attr:$slot, I32Attr:$byte_offset,
                       UnitAttr:$transient);

  let assemblyFormat = [{
    $sym_name (`transient` $transient^)? `slot` $slot `offset` $byte_offset
      attr-dict `:` $type
  }];
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
  let results = (outs AnyTypeOf<[Sol_RefType, Sol_MappingType]>:$addr);

  let builders = [
    OpBuilder<(ins "Value":$baseAddr, "Value":$idx)>
  ];

  let extraClassDeclaration = [{
    /// Returns the result type for a `sol.gep` whose base has type
    /// `baseAddrTy` and whose pointee is `elementType`.
    static Type getResultType(Type baseAddrTy, Type elementType);
  }];

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

def Sol_DeleteOp : Sol_Op<"delete"> {
  let summary = "Resets a reference-typed storage variable to its zero value.";
  let description = [{
    Recursively clears every storage slot occupied by the reference-typed value
    rooted at the given storage reference (Solidity `delete x`). Mirrors solc's
    deep storage clear: dynamic arrays/strings reset to empty (length slot zeroed,
    data tail freed); fixed arrays and structs recurse into their elements/members.
    The operand is a `Storage` reference, e.g. the result of `sol.addr_of` for an
    aggregate state variable.
  }];
  let arguments = (ins AnyType:$reference);

  let assemblyFormat = [{
    $reference attr-dict `:` type($reference)
  }];
}

// TODO: Simplify the asm format by inferring element type of Sol_RefType.

def Sol_LoadOp : Sol_Op<"load", [
  DeclareOpInterfaceMethods<ConditionallySpeculatable>,
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

  let assemblyFormat = "$name attr-dict `:` qualified(type($val))";
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

def Sol_FixedBytes4SelectorType : Type<
  CPred<"::llvm::isa<::mlir::sol::FixedBytesType>($_self) && "
        "::llvm::cast<::mlir::sol::FixedBytesType>($_self).getSize() == 4">,
  "!sol.fixedbytes<4>">;

def Sol_EncodeOp : Sol_Op<"encode", [AttrSizedOperandSegments]> {
  // When $packed is set, values are encoded without per-element padding.
  // When $selector is set, it is prepended as the first 4 bytes while $ins are
  // ABI-encoded as a tuple.
  let arguments = (ins Variadic<AnyType>:$ins,
                       Optional<Sol_FixedBytes4SelectorType>:$selector,
                       UnitAttr:$packed);
  // FIXME: AnyType -> Sol_StringType causes the asm printer to miss the prefix!
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
    (`selector` `(` $selector^ `)`)? ($ins^)?
    `:` type($selector) type($ins)
    `:` type($res) attr-dict
  }];
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
  let arguments = (ins I1:$cond, OptionalAttr<StrAttr>:$msg,
                       Variadic<AnyType>:$args, UnitAttr:$call);

  let assemblyFormat = [{
    $cond (`,` $msg^)? `(` ($args^ `:` type($args))? `)` attr-dict
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

def Sol_DefaultCallDataOp : Sol_Op<"default_calldata", [Pure]> {
  let summary = "Default value for a calldata aggregate type";
  let results = (outs Sol_AggType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Type":$type), [{
      $_state.addTypes(type);
    }]>
  ];

  let assemblyFormat = "attr-dict `:` type($result)";
}

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

def Sol_ExtFuncConstantOp : Sol_Op<"ext_func_constant", [Pure]> {
  let arguments = (ins Sol_AddressType:$addr, I32Attr:$selector);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$addr attr-dict `:` type($addr) `->` type($result)";
}

def Sol_ExtFuncSelectorOp : Sol_Op<"ext_func_selector", [Pure]> {
  let summary = "Extracts the selector from an external function reference";
  let arguments = (ins Sol_ExtFuncRefType:$func);
  let results = (outs Sol_FixedBytes4SelectorType:$result);
  let assemblyFormat = "$func attr-dict `:` qualified(type($func)) `->` type($result)";
}

def Sol_ExtFuncAddrOp : Sol_Op<"ext_func_addr", [Pure]> {
  let summary = "Extracts the address from an external function reference";
  let arguments = (ins Sol_ExtFuncRefType:$func);
  let results = (outs Sol_AddressType:$result);
  let assemblyFormat = "$func attr-dict `:` qualified(type($func)) `->` type($result)";
}

// Base class for indirect call ops (internal and external function pointers).
class Sol_ICallOpBase<string mnemonic, dag extraArgs = (ins),
                      string extraAsmFmt = "">
    : Sol_Op<mnemonic, [CallOpInterface]> {
  let arguments = !con((ins AnyType:$callee, Variadic<AnyType>:$callee_operands),
                       extraArgs,
                       (ins OptionalAttr<DictArrayAttr>:$arg_attrs,
                            OptionalAttr<DictArrayAttr>:$res_attrs));
  let results = (outs Variadic<AnyType>:$outs);

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      auto ops = getCalleeOperands();
      return {ops.begin(), ops.end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
    }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, cast<Value>(callee));
    }
  }];

  let assemblyFormat = !strconcat(
    "$callee `(` $callee_operands `)` ", extraAsmFmt,
    "attr-dict `:` type($callee) `,` functional-type($callee_operands, results)");
}

// Indirect call for internal function pointers.
def ICallOp : Sol_ICallOpBase<"icall">;

// Indirect call for external function pointers.
def ExtICallOp : Sol_ICallOpBase<"ext_icall",
    (ins UI256:$gas, UI256:$value, UnitAttr:$static_call, UnitAttr:$try_call),
    "`gas` $gas `value` $value ">;

def CallOp : Sol_Op<"call", [CallOpInterface]> {
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$outs);

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
  let results = (outs Sol_ContractType:$addr);

  let assemblyFormat = "attr-dict `:` qualified(type($addr))";
}

def Sol_CallerOp : Sol_Op<"caller"> {
  let summary = "Returns the caller address";
  let results = (outs Sol_AddressType:$addr);

  let assemblyFormat = "attr-dict `:` qualified(type($addr))";
}

def Sol_OriginOp : Sol_Op<"origin"> {
  let summary =
      "Returns the address of the original transaction sender (tx.origin)";
  let results = (outs Sol_AddressType:$addr);

  let assemblyFormat = "attr-dict `:` qualified(type($addr))";
}

def Sol_GasPriceOp : Sol_Op<"gasprice"> {
  let summary =
      "Returns the gas price of the current transaction (tx.gasprice)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_CallValueOp : Sol_Op<"callvalue"> {
  let summary =
      "Returns the amount of wei sent with the current call (msg.value)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_SigOp : Sol_Op<"sig"> {
  let summary = "Returns the first four bytes of the calldata (msg.sig)";
  let results = (outs Sol_FixedBytesType:$val);

  let assemblyFormat = "attr-dict `:` qualified(type($val))";
}

def Sol_BaseFeeOp : Sol_Op<"basefee"> {
  let summary = "Returns the current block's base fee (block.basefee)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_BlobBaseFeeOp : Sol_Op<"blobbasefee"> {
  let summary = "Returns the current block's blob base fee (block.blobbasefee)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_ChainIdOp : Sol_Op<"chainid"> {
  let summary = "Returns the current chain id (block.chainid)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_CoinbaseOp : Sol_Op<"coinbase"> {
  let summary = "Returns the current block miner's address (block.coinbase)";
  let results = (outs Sol_AddressType:$addr);

  let assemblyFormat = "attr-dict `:` qualified(type($addr))";
}

def Sol_DifficultyOp : Sol_Op<"difficulty"> {
  let summary = "Returns the current block difficulty; deprecated alias for "
                "block.prevrandao";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_GasLimitOp : Sol_Op<"gaslimit"> {
  let summary = "Returns the current block gas limit (block.gaslimit)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_BlockNumberOp : Sol_Op<"blocknumber"> {
  let summary = "Returns the current block number (block.number)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_PrevRandaoOp : Sol_Op<"prevrandao"> {
  let summary = "Returns the random number provided by the beacon chain "
                "(block.prevrandao)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_TimestampOp : Sol_Op<"timestamp"> {
  let summary = "Returns the current block timestamp in seconds since Unix "
                "epoch (block.timestamp)";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_GasLeftOp : Sol_Op<"gasleft"> {
  let summary = "Returns the remaining gas (gasleft())";
  let results = (outs UI256:$val);

  let assemblyFormat = "attr-dict `:` type($val)";
}

def Sol_BlockHashOp : Sol_Op<"blockhash"> {
  let summary = "Returns the hash of the given block (blockhash(uint))";
  let arguments = (ins UI256:$blockNumber);
  let results = (outs Sol_FixedBytesType:$val);

  let assemblyFormat = "$blockNumber attr-dict `:` type($val)";
}

def Sol_BlobHashOp : Sol_Op<"blobhash"> {
  let summary = "Returns the hash of the versioned blob (blobhash(uint))";
  let arguments = (ins UI256:$idx);
  let results = (outs Sol_FixedBytesType:$val);

  let assemblyFormat = "$idx attr-dict `:` type($val)";
}

def Sol_SelfdestructOp : Sol_Op<"selfdestruct"> {
  let summary = "Terminates the contract and sends balance to the recipient";
  let arguments = (ins Sol_AddressType:$recipient);

  let assemblyFormat = "$recipient attr-dict `:` qualified(type($recipient))";
}

def Sol_ExtCallOp : Sol_Op<"ext_call", [CallOpInterface]> {
  let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$ins,
                       AnyTypeOf<[Sol_AddressType, UI256]>:$addr,
                       UI256:$gas, UI256:$val,
                       UI256:$selector,
                       UnitAttr:$try_call, UnitAttr:$static_call,
                       UnitAttr:$delegate_call,
                       UnitAttr:$library_call,
                       TypeAttrOf<FunctionType>:$callee_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs I1:$status, Variadic<AnyType>:$outs);

  // TODO: Custom builder that doesn't ask for arg/res attrs.

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//
    operand_range getArgOperands() {
      auto ins = getIns();
      return {ins.begin(), ins.end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInsMutable();
    }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $ins `)` `at` $addr `gas` $gas `value` $val `selector` $selector
      attr-dict `:` qualified(type($addr)) `,` functional-type($ins, results)
  }];
}

def Sol_MemoryStringType : Type<
    CPred<"::llvm::isa<::mlir::sol::StringType>($_self) && "
          "::llvm::cast<::mlir::sol::StringType>($_self).getDataLocation() == "
          "::mlir::sol::DataLocation::Memory">,
    "!sol.string<Memory>">;

def Sol_BareCallOp : Sol_Op<"bare_call"> {
  let arguments = (ins Sol_AddressType:$addr, UI256:$gas, UI256:$val,
                       Sol_MemoryStringType:$inp);
  let results = (outs I1:$status, Sol_MemoryStringType:$ret_data);

  let assemblyFormat = [{
    $addr `gas` $gas `value` $val `input` $inp attr-dict `:`
      qualified(type($addr)) `,` type($gas) `,` type($val) `,`
      qualified(type($inp)) `->` type($status) `,`
      qualified(type($ret_data))
  }];
}

def Sol_BareDelegateCallOp : Sol_Op<"bare_delegate_call"> {
  let arguments = (ins Sol_AddressType:$addr, UI256:$gas,
                       Sol_MemoryStringType:$inp);
  let results = (outs I1:$status, Sol_MemoryStringType:$ret_data);

  let assemblyFormat = [{
    $addr `gas` $gas `input` $inp attr-dict `:`
      qualified(type($addr)) `,` type($gas) `,` qualified(type($inp))
      `->` type($status) `,` qualified(type($ret_data))
  }];
}

def Sol_BareStaticCallOp : Sol_Op<"bare_static_call"> {
  let arguments = (ins Sol_AddressType:$addr, UI256:$gas,
                       Sol_MemoryStringType:$inp);
  let results = (outs I1:$status, Sol_MemoryStringType:$ret_data);

  let assemblyFormat = [{
    $addr `gas` $gas `input` $inp attr-dict `:`
      qualified(type($addr)) `,` type($gas) `,` qualified(type($inp))
      `->` type($status) `,` qualified(type($ret_data))
  }];
}

def Sol_SendOp : Sol_Op<"send"> {
  let arguments = (ins Sol_AddressType:$addr, UI256:$val);
  let results = (outs I1:$status);

  let assemblyFormat = [{
    $addr `,` $val attr-dict `:` qualified(type($addr)) `,` type($val) `->` type($status)
  }];
}

def Sol_TransferOp : Sol_Op<"transfer"> {
  let arguments = (ins Sol_AddressType:$addr, UI256:$val);

  let assemblyFormat = [{
    $addr `,` $val attr-dict `:` qualified(type($addr)) `,` type($val)
  }];
}

def Sol_NewOp : Sol_Op<"new", [AttrSizedOperandSegments]> {
  let arguments = (ins StrAttr:$objName, UI256:$val,
                   Optional<UI256>:$salt,
                   Variadic<AnyType>:$ctorArgs,
                   UnitAttr:$try_call);
  let results = (outs Sol_ContractType:$out);

  let assemblyFormat = [{
    $objName `value` `=` $val (`salt` `=` $salt^)?
    `ctor` `(` ($ctorArgs^ `:` type($ctorArgs))? `)` (`try` $try_call^)? attr-dict
    `:` qualified(type($out))
  }];
}

def Sol_ObjectCodeOp : Sol_Op<"object_code"> {
  let arguments = (ins StrAttr:$objName);
  // FIXME: AnyType -> Sol_StringType causes the asm printer to miss the prefix!
  let results = (outs AnyType:$out);

  let assemblyFormat = [{
    $objName attr-dict `:` type($out)
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
    $contAddr attr-dict `:` qualified(type($contAddr)) `->` type($out)
  }];
}

def Sol_CodeHashOp : Sol_Op<"code_hash", [Pure]> {
  let arguments = (ins Sol_AddressType:$contAddr);
  let results = (outs UI256:$out);

  let assemblyFormat = [{
    $contAddr attr-dict `:` qualified(type($contAddr)) `->` type($out)
  }];
}

def Sol_BalanceOp : Sol_Op<"balance"> {
  let arguments = (ins Sol_AddressType:$contAddr);
  let results = (outs UI256:$out);

  let assemblyFormat = [{
    $contAddr attr-dict `:` qualified(type($contAddr)) `->` type($out)
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

def Sol_StringLitOp : Sol_Op<"string_lit"> {
  let arguments = (ins StrAttr:$value);
  let results = (outs Res<Sol_StringType, "", [MemAlloc]>:$addr);
  let assemblyFormat = "$value attr-dict `->` qualified(type($addr))";
}

def Sol_ConcatOp : Sol_Op<"concat"> {
  let summary = "Concatenate bytes/string values into a new memory string";
  let arguments =
      (ins Variadic<AnyTypeOf<[Sol_FixedBytesType, Sol_StringType]>>:$args);
  let results = (outs Sol_StringType:$result);

  let assemblyFormat = "($args^ `:` type($args))? attr-dict `->` type($result)";
}

#endif // MLIR_DIALECT_SOL_SOLOPS_TD

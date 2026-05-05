//===- YulOps.td - Yul dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_YUL_YULOPS_TD
#define MLIR_DIALECT_YUL_YULOPS_TD

include "YulBase.td"
include "YulInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def Yul_FuncOp : Yul_Op<"func", [AutomaticAllocationScope,
                                  CallableOpInterface,
                                  FunctionOpInterface,
                                  IsolatedFromAbove, Symbol]> {
  let summary = "Represents a Yul function";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type), [{
      $_state.addRegion();
      $_state.addAttribute(getSymNameAttrName($_state.name),
                           $_builder.getStringAttr(name));
      $_state.addAttribute(getFunctionTypeAttrName($_state.name),
                           TypeAttr::get(type));
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
  }];

  let assemblyFormat = "$sym_name `:` $function_type $body attr-dict";
}

def Yul_FuncCallOp : Yul_Op<"func_call"> {
  let summary = "Represents a Yul function call";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<I256>:$operands);
  let results = (outs Variadic<I256>:$outs);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute(getCalleeAttrName($_state.name),
                           SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Yul_FuncReturnOp : Yul_Op<"func_return", [Pure, Terminator]> {
  let summary = "Represents a Yul function return";
  let arguments = (ins Variadic<I256>:$operands);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, {});
    }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^)?";
}

def Yul_ConstantOp : Yul_Op<"constant", [
  Pure,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "Represents a Yul word constant";
  let arguments = (ins I256Attr:$value);
  let results = (outs I256:$out);

  let assemblyFormat = "$value attr-dict";
}

class Yul_BinOp<string mnemonic> :
    Yul_Op<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "out"]>]> {
  let arguments = (ins I256:$lhs, I256:$rhs);
  let results = (outs I256:$out);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Yul_AddOp : Yul_BinOp<"add"> {
  let summary = "Represents the `add` call in yul";
}

def Yul_SubOp : Yul_BinOp<"sub"> {
  let summary = "Represents the `sub` call in yul";
}

def Yul_MulOp : Yul_BinOp<"mul"> {
  let summary = "Represents the `mul` call in yul";
}

def Yul_AndOp : Yul_BinOp<"and"> {
  let summary = "Represents the `and` call in yul";
}

def Yul_OrOp : Yul_BinOp<"or"> {
  let summary = "Represents the `or` call in yul";
}

def Yul_XOrOp : Yul_BinOp<"xor"> {
  let summary = "Represents the `xor` call in yul";
}

def Yul_NotOp : Yul_Op<"not", [Pure, AllTypesMatch<["value", "out"]>]> {
  let summary = "Represents the `not` call in yul";
  let arguments = (ins I256:$value);
  let results = (outs I256:$out);
  let assemblyFormat = "$value attr-dict";
}

def Yul_CmpOp : Yul_Op<"cmp", [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "Represents a Yul word comparison";
  let arguments = (ins Yul_CmpPredicateAttr:$predicate,
                       I256:$lhs, I256:$rhs);
  let results = (outs I256:$out);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict";
}

class Yul_ArithBinOp<string mnemonic> :
    Yul_Op<mnemonic, [Pure, AllTypesMatch<["lhs", "rhs", "out"]>]> {
  let arguments = (ins I256:$lhs, I256:$rhs);
  let results = (outs I256:$out);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Yul_ArithShlOp : Yul_ArithBinOp<"arith_shl"> {
  let summary = "UB-prone arithmetic wrapper for logical left shift; all inputs are i256";
}

def Yul_ArithShrOp : Yul_ArithBinOp<"arith_shr"> {
  let summary = "UB-prone arithmetic wrapper for logical right shift; all inputs are i256";
}

def Yul_ArithSarOp : Yul_ArithBinOp<"arith_sar"> {
  let summary = "UB-prone arithmetic wrapper for arithmetic right shift; all inputs are i256";
}

def Yul_ArithDivOp : Yul_ArithBinOp<"arith_div"> {
  let summary = "UB-prone arithmetic wrapper for unsigned division; all inputs are i256";
}

def Yul_ArithSDivOp : Yul_ArithBinOp<"arith_sdiv"> {
  let summary = "UB-prone arithmetic wrapper for signed division; all inputs are i256";
}

def Yul_ArithModOp : Yul_ArithBinOp<"arith_mod"> {
  let summary = "UB-prone arithmetic wrapper for unsigned remainder; all inputs are i256";
}

def Yul_ArithSModOp : Yul_ArithBinOp<"arith_smod"> {
  let summary = "UB-prone arithmetic wrapper for signed remainder; all inputs are i256";
}

def Yul_ArithSelectOp : Yul_Op<"arith_select", [Pure,
    AllTypesMatch<["trueValue", "falseValue", "out"]>]> {
  let summary = "Arithmetic wrapper for value selection; all inputs are i256";
  let arguments = (ins I256:$condition, I256:$trueValue, I256:$falseValue);
  let results = (outs I256:$out);
  let assemblyFormat =
      "$condition `,` $trueValue `,` $falseValue attr-dict";
}

def Yul_BreakOp : Yul_Op<"break", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Yul_ContinueOp : Yul_Op<"continue", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

def Yul_ConditionOp : Yul_Op<"condition", [Terminator]> {
  let arguments = (ins I256:$condition, Variadic<I256>:$args);

  let builders = [
    OpBuilder<(ins "Value":$condition, CArg<"ValueRange", "{}">:$args), [{
      $_state.addOperands(condition);
      $_state.addOperands(args);
    }]>,
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = "$condition (`,` $args^)? attr-dict";
}

def Yul_YieldOp : Yul_Op<"yield", [ReturnLike, Terminator,
  ParentOneOf<["IfOp", "SwitchOp", "WhileOp", "DoWhileOp", "ForOp"]>]
> {
  let arguments = (ins Variadic<I256>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "($operands^)? attr-dict";
}

def Yul_IfOp : Yul_Op<"if",
  [AutomaticAllocationScope, NoRegionArguments,
   DeclareOpInterfaceMethods<RegionBranchOpInterface>]
> {
  let arguments = (ins I256:$cond);
  let results = (outs Variadic<I256>:$results);
  let regions = (region MinSizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let builders = [
    OpBuilder<(ins "Value":$cond), [{
      $_state.addOperands(cond);
      $_state.addRegion();
      $_state.addRegion();
    }]>,
  ];

  let assemblyFormat =
      "$cond (`->` type($results)^)? $thenRegion `else` $elseRegion attr-dict";
}

def Yul_SwitchOp : Yul_Op<"switch", [
  AutomaticAllocationScope, NoRegionArguments,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                            ["getRegionInvocationBounds"]>
]> {
  let arguments = (ins I256:$arg, SignlessIntElementsAttr<256>:$cases);
  let results = (outs Variadic<I256>:$results);
  let regions = (region MinSizedRegion<1>:$defaultRegion,
                        VariadicRegion<MinSizedRegion<1>>:$caseRegions);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

class Yul_WhileOpBase<string mnemonic> : Yul_Op<mnemonic, [
  Yul_LoopOpInterface, AutomaticAllocationScope
]>;

def Yul_WhileOp : Yul_WhileOpBase<"while"> {
  let arguments = (ins Variadic<I256>:$inits);
  let results = (outs Variadic<I256>:$results);
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body);

  let assemblyFormat =
      "($inits^)? (`->` type($results)^)? $cond `do` $body attr-dict";
}

def Yul_DoWhileOp : Yul_WhileOpBase<"do"> {
  let regions = (region MinSizedRegion<1>:$body, SizedRegion<1>:$cond);

  let assemblyFormat = "$body `while` $cond attr-dict";

  let extraClassDeclaration = [{
    Region &getEntry() { return getBody(); }
  }];
}

def Yul_ForOp : Yul_Op<"for", [Yul_LoopOpInterface,
                               AutomaticAllocationScope]> {
  let arguments = (ins Variadic<I256>:$inits,
                       OptionalAttr<UnitAttr>:$full_unroll);
  let results = (outs Variadic<I256>:$results);
  let regions = (region SizedRegion<1>:$cond, MinSizedRegion<1>:$body,
                        SizedRegion<1>:$step);

  let builders = [
    OpBuilder<(ins CArg<"ValueRange", "{}">:$inits,
                   CArg<"UnitAttr", "{}">:$full_unroll), [{
      $_state.addOperands(inits);
      if (full_unroll)
        $_state.addAttribute(getFullUnrollAttrName($_state.name), full_unroll);
      $_state.addTypes(TypeRange(inits));
      $_state.addRegion();
      $_state.addRegion();
      $_state.addRegion();
    }]>,
  ];

  let assemblyFormat = [{
    ($inits^)? (`->` type($results)^)?
    `cond` $cond
    `body` $body
    `step` $step
    attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::Region *maybeGetStep() { return &getStep(); }
  }];
}

def Yul_UpdFreePtrOp : Yul_Op<"upd_free_ptr"> {
  let arguments = (ins I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

// FIXME: Wrap the entry code to an entry function so that we can add the
// SingleBlock trait.
def Yul_ObjectOp : Yul_Op<"object", [Symbol, SymbolTable, NoTerminator,
                                     IsolatedFromAbove]> {
  let summary = "Represents an `object` in yul";

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins CArg<"StringRef", "{}">:$name),[{
      $_state.addRegion()->emplaceBlock();
      $_state.addAttribute(getSymNameAttrName($_state.name),
                           $_builder.getStringAttr(name));
    }]>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    Block *getEntryBlock() { return &getBody().front(); }
  }];

  let assemblyFormat = "$sym_name $body attr-dict";
}

//
// We could auto-generate the following evm builtin ops from
// evmasm::InstructionInfo. But I don't think it's worth the effort.
//

def Yul_CallerOp : Yul_Op<"caller", [Pure]> {
  let summary = "Represents the `caller` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_GasOp : Yul_Op<"gas", [Pure]> {
  let summary = "Represents the `gas` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_AddressOp : Yul_Op<"address", [Pure]> {
  let summary = "Represents the `address` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_BalanceOp : Yul_Op<"balance"> {
  let summary = "Represents the `balance` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_SelfBalanceOp : Yul_Op<"selfbalance"> {
  let summary = "Represents the `selfbalance` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallOp : Yul_Op<"call"> {
  let summary = "Represents the `call` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$value, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$status);
  let assemblyFormat = [{
    $gas `,` $address `,` $value `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_CallCodeOp : Yul_Op<"callcode"> {
  let summary = "Represents the `callcode` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$value, I256:$inpOffset,
      I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$status);
  let assemblyFormat = [{
    $gas `,` $address `,` $value `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_StaticCallOp : Yul_Op<"static_call"> {
  let summary = "Represents the `staticcall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_DelegateCallOp : Yul_Op<"delegate_call"> {
  let summary = "Represents the `delegatecall` call in yul";
  let arguments = (ins I256:$gas, I256:$address, I256:$inpOffset,
                       I256:$inpSize, I256:$outOffset, I256:$outSize);
  let results = (outs I256:$out);
  let assemblyFormat = [{
    $gas `,` $address `,` $inpOffset `,` $inpSize `,` $outOffset `,`
    $outSize attr-dict
  }];
}

def Yul_ReturnOp : Yul_Op<"return"> {
  let summary = "Represents the `return` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_RevertOp : Yul_Op<"revert"> {
  let summary = "Represents the `revert` call in yul";
  // TODO: Support revert strings.
  let arguments = (ins I256:$addr, I256:$size);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_StopOp : Yul_Op<"stop"> {
  let summary = "Represents the `stop` call in yul";
  let assemblyFormat = "attr-dict";
}

def Yul_InvalidOp : Yul_Op<"invalid"> {
  let summary = "Represents the `invalid` call in yul";
  let assemblyFormat = "attr-dict";
}

def Yul_MLoadOp : Yul_Op<"mload"> {
  let summary = "Represents the `mload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_MStoreOp : Yul_Op<"mstore"> {
  let summary = "Represents the `mstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_MStore8Op : Yul_Op<"mstore8"> {
  let summary = "Represents the `mstore8` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_ByteOp : Yul_Op<"byte"> {
  let summary = "Represents the `byte` call in yul";
  let arguments = (ins I256:$idx, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$idx `,` $val attr-dict";
}

def Yul_MCopyOp : Yul_Op<"mcopy"> {
  let summary = "Represents the `mcopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_MSizeOp : Yul_Op<"msize"> {
  let summary = "Represents the `msize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallValOp : Yul_Op<"callvalue", [Pure]> {
  let summary = "Represents the `callvalue` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallDataLoadOp : Yul_Op<"calldataload"> {
  let summary = "Represents the `calldataload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_CallDataSizeOp : Yul_Op<"calldatasize", [Pure]> {
  let summary = "Represents the `calldatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CallDataCopyOp : Yul_Op<"calldatacopy"> {
  let summary = "Represents the `calldatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_ReturnDataSizeOp : Yul_Op<"returndatasize", [Pure]> {
  let summary = "Represents the `returndatasize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_ReturnDataCopyOp : Yul_Op<"returndatacopy"> {
  let summary = "Represents the `returndatacopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_SLoadOp : Yul_Op<"sload"> {
  let summary = "Represents the `sload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_SStoreOp : Yul_Op<"sstore"> {
  let summary = "Represents the `sstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

def Yul_TLoadOp : Yul_Op<"tload"> {
  let summary = "Represents the `tload` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_TStoreOp : Yul_Op<"tstore"> {
  let summary = "Represents the `tstore` call in yul";
  let arguments = (ins I256:$addr, I256:$val);
  let assemblyFormat = "$addr `,` $val attr-dict";
}

// TODO: Support symbolic references to objects outside the current symbol table
// (including ones outside the translation unit) using SymbolRefAttr instead of
// FlatSymbolRefAttr

def Yul_DataOffsetOp : Yul_Op<"dataoffset", [Pure]> {
  let summary = "Represents the `dataoffset` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_DataSizeOp : Yul_Op<"datasize", [Pure]> {
  let summary = "Represents the `datasize` call in yul";
  let arguments = (ins FlatSymbolRefAttr:$obj);
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CodeSizeOp : Yul_Op<"codesize", [Pure]> {
  let summary = "Represents the `codesize` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_CodeCopyOp : Yul_Op<"codecopy"> {
  let summary = "Represents the `codecopy` call in yul";
  let arguments = (ins I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$dst `,` $src `,` $size attr-dict";
}

def Yul_ExtCodeCopyOp : Yul_Op<"extcodecopy"> {
  let summary = "Represents the `extcodecopy` call in yul";
  let arguments = (ins I256:$addr, I256:$dst, I256:$src, I256:$size);
  let assemblyFormat = "$addr `,` $dst `,` $src `,` $size attr-dict";
}

def Yul_ExtCodeSizeOp : Yul_Op<"extcodesize", [Pure]> {
  let summary = "Represents the `extcodesize` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_ExtCodeHashOp : Yul_Op<"extcodehash", [Pure]> {
  let summary = "Represents the `extcodehash` call in yul";
  let arguments = (ins I256:$addr);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr attr-dict";
}

def Yul_CreateOp : Yul_Op<"create"> {
  let summary = "Represents the `create` call in yul";
  let arguments = (ins I256:$val, I256:$addr, I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$val `,` $addr `,` $size attr-dict";
}

def Yul_Create2Op : Yul_Op<"create2"> {
  let summary = "Represents the `create2` call in yul";
  let arguments = (ins I256:$val, I256:$addr, I256:$size, I256:$salt);
  let results = (outs I256:$out);
  let assemblyFormat = "$val `,` $addr `,` $size `,` $salt attr-dict";
}

def Yul_ChainIdOp : Yul_Op<"chainid"> {
  let summary = "Represents the `chainid` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_BaseFeeOp : Yul_Op<"basefee"> {
  let summary = "Represents the `basefee` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_BlobBaseFeeOp : Yul_Op<"blobbasefee"> {
  let summary = "Represents the `blobbasefee` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_OriginOp : Yul_Op<"origin"> {
  let summary = "Represents the `origin` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_GasPriceOp : Yul_Op<"gasprice"> {
  let summary = "Represents the `gasprice` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_BlockHashOp : Yul_Op<"blockhash"> {
  let summary = "Represents the `blockhash` call in yul";
  let arguments = (ins I256:$block);
  let results = (outs I256:$out);
  let assemblyFormat = "$block attr-dict";
}

def Yul_BlobHashOp : Yul_Op<"blobhash"> {
  let summary = "Represents the `blobhash` call in yul";
  let arguments = (ins I256:$Idx);
  let results = (outs I256:$out);
  let assemblyFormat = "$Idx attr-dict";
}

def Yul_CoinBaseOp : Yul_Op<"coinbase"> {
  let summary = "Represents the `coinbase` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_TimeStampOp : Yul_Op<"timestamp"> {
  let summary = "Represents the `timestamp` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_NumberOp : Yul_Op<"number"> {
  let summary = "Represents the `number` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_PrevrandaoOp : Yul_Op<"prevrandao"> {
  let summary = "Represents the `prevrandao` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

def Yul_GasLimitOp : Yul_Op<"gaslimit"> {
  let summary = "Represents the `gaslimit` call in yul";
  let results = (outs I256:$out);
  let assemblyFormat = "attr-dict";
}

// TODO: Is this `ConstantLike`? Adding it causes "expected ConstantLike op to
// be foldable" assert fail (probably due to the missing let hasFolder = 1).
def Yul_MemGuardOp : Yul_Op<"memoryguard", [Pure]> {
  let summary = "Represents the `memoryguard` call in yul";
  let arguments = (ins I256Attr:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$size attr-dict";
}

def Yul_Keccak256Op : Yul_Op<"keccak256"> {
  // Note that the keccak256 yul builtin has side effects.
  let summary = "Represents the `keccak256` call in yul";
  let arguments = (ins I256:$addr, I256:$size);
  let results = (outs I256:$out);
  let assemblyFormat = "$addr `,` $size attr-dict";
}

def Yul_DivOp : Yul_Op<"div", [Pure]> {
  let summary = "Represents the `div` call in yul";
  let arguments = (ins I256:$dividend, I256:$divisor);
  let results = (outs I256:$out);
  let assemblyFormat = "$dividend `,` $divisor attr-dict";
}

def Yul_SDivOp : Yul_Op<"sdiv", [Pure]> {
  let summary = "Represents the `sdiv` call in yul";
  let arguments = (ins I256:$dividend, I256:$divisor);
  let results = (outs I256:$out);
  let assemblyFormat = "$dividend `,` $divisor attr-dict";
}

def Yul_ModOp : Yul_Op<"mod", [Pure]> {
  let summary = "Represents the `mod` call in yul";
  let arguments = (ins I256:$value, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$value `,` $mod attr-dict";
}

def Yul_SModOp : Yul_Op<"smod", [Pure]> {
  let summary = "Represents the `smod` call in yul";
  let arguments = (ins I256:$value, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$value `,` $mod attr-dict";
}

def Yul_ShlOp : Yul_Op<"shl", [Pure]> {
  let summary = "Represents the `shl` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_ShrOp : Yul_Op<"shr", [Pure]> {
  let summary = "Represents the `shr` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_SarOp : Yul_Op<"sar", [Pure]> {
  let summary = "Represents the `sar` call in yul";
  let arguments = (ins I256:$shift, I256:$val);
  let results = (outs I256:$out);
  let assemblyFormat = "$shift `,` $val attr-dict";
}

def Yul_ExpOp : Yul_Op<"exp", [Pure]> {
  let summary = "Represents the `exp` call in yul";
  let arguments = (ins I256:$base, I256:$exp);
  let results = (outs I256:$out);
  let assemblyFormat = "$base `,` $exp attr-dict";
}

def Yul_AddModOp : Yul_Op<"addmod", [Pure]> {
  let summary = "Represents the `addmod` call in yul";
  let arguments = (ins I256:$x, I256:$y, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$x `,` $y `,` $mod attr-dict";
}

def Yul_MulModOp : Yul_Op<"mulmod", [Pure]> {
  let summary = "Represents the `mulmod` call in yul";
  let arguments = (ins I256:$x, I256:$y, I256:$mod);
  let results = (outs I256:$out);
  let assemblyFormat = "$x `,` $y `,` $mod attr-dict";
}

def Yul_LogOp : Yul_Op<"log"> {
  let summary = "Represents the `log*` calls in yul";
  let arguments = (ins I256:$addr, I256:$size, Variadic<I256>:$topics);
  // FIXME: Trailing spaces with empty $topics.
  let assemblyFormat = "$addr `,` $size oilist(`topics` `(` $topics `)` ) attr-dict";
}

def Yul_SignExtendOp : Yul_Op<"signextend", [Pure]> {
  let summary = "Represents the `signextend` call in yul";
  let arguments = (ins I256:$val, I256:$off);
  let results = (outs I256:$out);
  let assemblyFormat = "$val `,` $off attr-dict";
}

def Yul_SetImmutableOp : Yul_Op<"setimmutable"> {
  let summary = "Represents the `setimmutable` call in yul";
  let arguments = (ins I256:$addr, StrAttr:$name, I256:$val);
  let assemblyFormat = "$addr `,` $name `,` $val attr-dict";
}

def Yul_LoadImmutableOp : Yul_Op<"loadimmutable"> {
  let summary = "Represents the `loadimmutable` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

def Yul_LinkerSymbolOp : Yul_Op<"linkersymbol"> {
  let summary = "Represents the `linkersymbol` call in yul";
  let arguments = (ins StrAttr:$name);
  let results = (outs I256:$out);
  let assemblyFormat = "$name attr-dict";
}

#endif // MLIR_DIALECT_YUL_YULOPS_TD

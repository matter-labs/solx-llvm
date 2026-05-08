//===- YulBase.td - Yul dialect ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_YUL_YULBASE_TD
#define MLIR_DIALECT_YUL_YULBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Yul_Dialect : Dialect {
  let name = "yul";
  let cppNamespace = "::mlir::yul";
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def I256 : I<256>;

def I256OrI256LLVMStruct : Type<
  CPred<"::mlir::yul::isI256OrI256LLVMStruct($_self)">,
  "i256 or LLVM struct type containing only i256 elements">;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def I256Attr : SignlessIntegerAttrBase<I256,
                                       "256-bit signless integer attribute">;

def Yul_CmpPredicateAttr :
    I64EnumAttr<"CmpPredicate", "Yul comparison predicate",
                [I64EnumAttrCase<"eq", 0>,
                 I64EnumAttrCase<"ne", 1>,
                 I64EnumAttrCase<"ult", 2>,
                 I64EnumAttrCase<"ule", 3>,
                 I64EnumAttrCase<"ugt", 4>,
                 I64EnumAttrCase<"uge", 5>,
                 I64EnumAttrCase<"slt", 6>,
                 I64EnumAttrCase<"sle", 7>,
                 I64EnumAttrCase<"sgt", 8>,
                 I64EnumAttrCase<"sge", 9>]> {
  let cppNamespace = "::mlir::yul";
}

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class Yul_Op<string mnemonic, list<Trait> traits = []> : Op<Yul_Dialect,
                                                            mnemonic, traits>;

#endif // MLIR_DIALECT_YUL_YULBASE_TD

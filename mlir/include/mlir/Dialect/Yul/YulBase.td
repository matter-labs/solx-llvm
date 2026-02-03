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

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def I256Attr : SignlessIntegerAttrBase<I256,
                                       "256-bit signless integer attribute">;

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class Yul_Op<string mnemonic, list<Trait> traits = []> : Op<Yul_Dialect,
                                                            mnemonic, traits>;

#endif // MLIR_DIALECT_YUL_YULBASE_TD

//===- Sol.cpp - C API for Sol dialect ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Sol.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/Dialect/Sol/Transforms/SolImmutables.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sol, sol, mlir::sol::SolDialect)

void mlirEvmLowerSetImmutables(MlirModule mod, const char **immIDs,
                               const uint64_t *immOffsets, uint64_t immCount) {
  llvm::StringMap<mlir::SmallVector<uint64_t>> immMap;
  for (uint64_t i = 0; i < immCount; ++i)
    immMap[immIDs[i]].push_back(immOffsets[i]);
  mlir::evm::lowerSetImmutables(unwrap(mod), immMap);
}

void mlirEvmRemoveSetImmutables(MlirModule mod) {
  mlir::evm::removeSetImmutables(unwrap(mod));
}

MlirType mlirSolGetEltType(MlirType ty, uint64_t structFieldIdx) {
  return wrap(mlir::sol::getEltType(unwrap(ty), structFieldIdx));
}

MlirType mlirSolGepGetResultType(MlirType baseAddrTy, MlirType elementType) {
  return wrap(
      mlir::sol::GepOp::getResultType(unwrap(baseAddrTy), unwrap(elementType)));
}

MlirValue mlirSolEmitZeroedVal(MlirBlock block, MlirOperation insertBefore,
                               MlirType ty, MlirLocation loc) {
  mlir::Block *blk = unwrap(block);
  mlir::OpBuilder b(unwrap(ty).getContext());
  if (!mlirOperationIsNull(insertBefore))
    b.setInsertionPoint(unwrap(insertBefore));
  else
    b.setInsertionPointToEnd(blk);
  return wrap(mlir::sol::emitZeroedVal(b, unwrap(ty), unwrap(loc)));
}

//===- sol.c - Test of Sol dialect C API ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-sol-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/Sol.h"
#include "mlir-c/Conversion.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#include <stdio.h>
#include <stdlib.h>

static void dontPrint(MlirStringRef str, void *userData) {
  (void)str;
  (void)userData;
}

// CHECK-LABEL: testSolDialectHandle
static void testSolDialectHandle(MlirContext ctx) {
  fprintf(stderr, "testSolDialectHandle\n");

  MlirDialectHandle solHandle = mlirGetDialectHandle__sol__();
  mlirDialectHandleRegisterDialect(solHandle, ctx);
  MlirDialect solDialect = mlirDialectHandleLoadDialect(solHandle, ctx);

  if (mlirDialectIsNull(solDialect)) {
    fprintf(stderr, "Sol dialect not loaded\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: Sol dialect loaded successfully
  fprintf(stderr, "Sol dialect loaded successfully\n");

  MlirStringRef solFunc = mlirStringRefCreateFromCString("sol.func");
  if (!mlirContextIsRegisteredOperation(ctx, solFunc)) {
    fprintf(stderr, "sol.func op not registered\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: sol.func op registered successfully
  fprintf(stderr, "sol.func op registered successfully\n");
}

// CHECK-LABEL: testSolPassRegistration
static void testSolPassRegistration(MlirContext ctx) {
  fprintf(stderr, "testSolPassRegistration\n");

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetAsOpPassManager(pm);
  MlirStringRef pipeline = mlirStringRefCreateFromCString(
      "builtin.module(sol-licm,sol-lower-modifier,convert-sol-to-std)");

  MlirLogicalResult status =
      mlirParsePassPipeline(opm, pipeline, dontPrint, NULL);
  if (mlirLogicalResultIsSuccess(status)) {
    fprintf(stderr, "Pass pipeline parsed before registration\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: parsing failed before registration
  fprintf(stderr, "parsing failed before registration\n");

  mlirRegisterSolPasses();
  mlirRegisterConversionConvertSolToStandardPass();

  status = mlirParsePassPipeline(opm, pipeline, dontPrint, NULL);
  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Pass pipeline parsing failed after registration\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: parsing succeeded after registration
  fprintf(stderr, "parsing succeeded after registration\n");

  mlirPassManagerDestroy(pm);
}

// CHECK-LABEL: testSolPasses
static void testSolPasses(MlirContext ctx) {
  fprintf(stderr, "testSolPasses\n");

  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirModule module = mlirModuleCreateEmpty(loc);
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  mlirPassManagerAddOwnedPass(pm, mlirCreateSolModifierOpLoweringPass());
  mlirPassManagerAddOwnedPass(pm, mlirCreateSolLoopInvariantCodeMotionPass());
  mlirPassManagerAddOwnedPass(pm,
                              mlirCreateConversionConvertSolToStandardPass());

  MlirLogicalResult status = mlirPassManagerRunOnOp(pm, moduleOp);
  if (!mlirLogicalResultIsSuccess(status)) {
    fprintf(stderr, "Pass manager run failed\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: pass manager run successfully
  fprintf(stderr, "pass manager run successfully\n");

  mlirEvmLowerSetImmutables(module, NULL, NULL, 0);
  mlirEvmRemoveSetImmutables(module);
  // CHECK: immutables run successfully
  fprintf(stderr, "immutables run successfully\n");

  mlirPassManagerDestroy(pm);
  mlirModuleDestroy(module);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  testSolDialectHandle(ctx);
  testSolPassRegistration(ctx);
  testSolPasses(ctx);
  mlirContextDestroy(ctx);
  return 0;
}

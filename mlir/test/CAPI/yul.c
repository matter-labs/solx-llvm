//===- yul.c - Test of Yul dialect C API ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-yul-test 2>&1 | FileCheck %s

#include "mlir-c/Dialect/Yul.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#include <stdio.h>
#include <stdlib.h>

static void dontPrint(MlirStringRef str, void *userData) {
  (void)userData;
  (void)str;
}

// CHECK-LABEL: testYulDialectHandle
static void testYulDialectHandle(MlirContext ctx) {
  fprintf(stderr, "testYulDialectHandle\n");

  MlirDialectHandle yulHandle = mlirGetDialectHandle__yul__();
  mlirDialectHandleRegisterDialect(yulHandle, ctx);
  MlirDialect yulDialect = mlirDialectHandleLoadDialect(yulHandle, ctx);

  if (mlirDialectIsNull(yulDialect)) {
    fprintf(stderr, "Yul dialect not loaded\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: Yul dialect loaded successfully
  fprintf(stderr, "Yul dialect loaded successfully\n");

  MlirStringRef yulCaller = mlirStringRefCreateFromCString("yul.caller");
  if (!mlirContextIsRegisteredOperation(ctx, yulCaller)) {
    fprintf(stderr, "yul.caller op not registered\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: yul.caller op registered successfully
  fprintf(stderr, "yul.caller op registered successfully\n");
}

// CHECK-LABEL: testYulPassRegistration
static void testYulPassRegistration(MlirContext ctx) {
  fprintf(stderr, "testYulPassRegistration\n");

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetAsOpPassManager(pm);
  MlirStringRef pipeline =
      mlirStringRefCreateFromCString("builtin.module(yul-fuse-free-ptr)");

  MlirLogicalResult status =
      mlirParsePassPipeline(opm, pipeline, dontPrint, NULL);
  if (mlirLogicalResultIsSuccess(status)) {
    fprintf(stderr, "Pass pipeline parsed before registration\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: parsing failed before registration
  fprintf(stderr, "parsing failed before registration\n");

  mlirRegisterYulPasses();

  status = mlirParsePassPipeline(opm, pipeline, dontPrint, NULL);
  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Pass pipeline parsing failed after registration\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: parsing succeeded after registration
  fprintf(stderr, "parsing succeeded after registration\n");

  mlirPassManagerDestroy(pm);
}

// CHECK-LABEL: testYulPasses
static void testYulPasses(MlirContext ctx) {
  fprintf(stderr, "testYulPasses\n");

  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirModule module = mlirModuleCreateEmpty(loc);
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirPassManager pm = mlirPassManagerCreate(ctx);
  mlirPassManagerAddOwnedPass(pm, mlirCreateYulFuseFreePtrPass());

  MlirLogicalResult status = mlirPassManagerRunOnOp(pm, moduleOp);
  if (!mlirLogicalResultIsSuccess(status)) {
    fprintf(stderr, "Pass manager run failed\n");
    exit(EXIT_FAILURE);
  }
  // CHECK: pass manager run successfully
  fprintf(stderr, "pass manager run successfully\n");

  mlirPassManagerDestroy(pm);
  mlirModuleDestroy(module);
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  testYulDialectHandle(ctx);
  testYulPassRegistration(ctx);
  testYulPasses(ctx);
  mlirContextDestroy(ctx);
  return 0;
}

//===-- EVMVerifier.cpp - EVM-specific IR verifier ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements EVM-specific IR verifier checks. It walks the whole
// module with an InstVisitor, reports each violation on dbgs() with the
// offending instruction, and aborts via report_fatal_error if the module is
// broken.
//
//===----------------------------------------------------------------------===//

#include "EVM.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "evm-verifier"

using namespace llvm;

namespace {
class EVMVerifierImpl : public InstVisitor<EVMVerifierImpl> {
public:
  EVMVerifierImpl(raw_ostream &OS, const Module &M) : OS(OS), MST(&M) {}

  /// Runs all EVM-specific checks over the module. Returns true if the module
  /// is well-formed, false if any violations were emitted.
  bool verify(Module &M);

  // InstVisitor hooks.
  void visitMemTransferInst(MemTransferInst &MTI);
  void visitMemSetInst(MemSetInst &MSI);

private:
  raw_ostream &OS;
  ModuleSlotTracker MST;
  bool Broken = false;

  void CheckFailed(const Twine &Message, const Value &V) {
    OS << Message << '\n';
    V.print(OS, MST);
    OS << '\n';
    Broken = true;
  }
};

class EVMVerifier final : public ModulePass {
public:
  static char ID;
  EVMVerifier() : ModulePass(ID) {}

  StringRef getPassName() const override { return "EVM Verifier"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override;
};

} // end anonymous namespace

// We know that cond should be true, if not print an error message. This is the
// same as in Verifier.cpp.
#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      CheckFailed(__VA_ARGS__);                                                \
      return;                                                                  \
    }                                                                          \
  } while (false)

void EVMVerifierImpl::visitMemTransferInst(MemTransferInst &MTI) {
  unsigned DestAS = MTI.getDestAddressSpace();
  unsigned SrcAS = MTI.getSourceAddressSpace();
  Check(DestAS == EVMAS::AS_HEAP,
        "EVM memcpy/memmove must write to heap memory", MTI);
  Check(SrcAS == EVMAS::AS_HEAP || SrcAS == EVMAS::AS_CALL_DATA ||
            SrcAS == EVMAS::AS_RETURN_DATA || SrcAS == EVMAS::AS_CODE,
        "EVM memcpy/memmove must read from heap, calldata, "
        "returndata, or code memory",
        MTI);
}

void EVMVerifierImpl::visitMemSetInst(MemSetInst &MSI) {
  Check(false, "EVM does not support generic memset intrinsics", MSI);
}

bool EVMVerifierImpl::verify(Module &M) {
  visit(M);
  return !Broken;
}

bool EVMVerifier::runOnModule(Module &M) {
  if (!EVMVerifierImpl(dbgs(), M).verify(M))
    report_fatal_error("Broken module found, compilation aborted!",
                       /*gen_crash_diag=*/false);
  return false;
}

char EVMVerifier::ID = 0;

INITIALIZE_PASS(EVMVerifier, DEBUG_TYPE, "EVM Verifier", false, false)

ModulePass *llvm::createEVMVerifierPass() { return new EVMVerifier(); }

PreservedAnalyses EVMVerifierPass::run(Module &M, ModuleAnalysisManager &) {
  if (!EVMVerifierImpl(dbgs(), M).verify(M))
    report_fatal_error("Broken module found, compilation aborted!",
                       /*gen_crash_diag=*/false);
  return PreservedAnalyses::all();
}

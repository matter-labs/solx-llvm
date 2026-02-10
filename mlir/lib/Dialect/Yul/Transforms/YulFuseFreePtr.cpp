//===- YulFuseFreePtr.cpp - Fuse free pointer operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Yul dialect free-ptr fusing pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Yul/Transforms/Passes.h"
#include "mlir/Dialect/Yul/Yul.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace yul {
#define GEN_PASS_DEF_FUSEFREEPTRPASS
#include "mlir/Dialect/Yul/Transforms/Passes.h.inc"
} // namespace yul
} // namespace mlir

using namespace mlir;

struct FuseFreePtrPass
    : public yul::impl::FuseFreePtrPassBase<FuseFreePtrPass> {
  FuseFreePtrPass() = default;
  FuseFreePtrPass(const FuseFreePtrPass &other) : FuseFreePtrPassBase(other) {}

  Statistic count{this, "count", "Number free-ptr updates fused"};

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto fnOp = dyn_cast<FunctionOpInterface>(op)) {
        for (Block &blk : fnOp.getFunctionBody()) {
          run(blk);
        }
      }
    });
  }

  void run(Block &blk) {
    if (blk.empty())
      return;

    SmallVector<yul::UpdFreePtrOp, 4> updOps;
    SmallPtrSet<Operation *, 4> updOpsUsers;

    auto fuseAndReset = [&]() {
      fuse(updOps);

      // Start over again.
      updOps.clear();
      updOpsUsers.clear();
    };

    for (Operation &op : llvm::make_early_inc_range(blk)) {
      if (updOpsUsers.contains(&op)) {
        fuseAndReset();
        continue;
      }
      if (auto updOp = dyn_cast<yul::UpdFreePtrOp>(op)) {
        for (Operation *user : updOp->getUsers())
          updOpsUsers.insert(user);
        updOps.push_back(updOp);
      }
    }
    fuse(updOps);
  }

  /// Fuses all the UpdFreePtrOps into one at the last op so that the use-graph
  /// is not broken. This expects the user-graph to be dominated by the last op.
  void fuse(ArrayRef<yul::UpdFreePtrOp> updOps) {
    if (updOps.size() < 2)
      return;

    // Generate the total size of allocated area after all the UpdFreePtrOps.
    count += updOps.size();
    yul::UpdFreePtrOp firstUpd = updOps.front();
    IRRewriter r(firstUpd);
    SmallVector<Location> fusedLocs{firstUpd.getLoc()};
    Value totalSize = firstUpd.getSize();
    for (yul::UpdFreePtrOp updOp : llvm::drop_begin(updOps)) {
      fusedLocs.push_back(updOp.getLoc());
      r.setInsertionPoint(updOp);
      totalSize =
          r.create<arith::AddIOp>(updOp.getLoc(), updOp.getSize(), totalSize);
    }

    // Generate the fused UpdFreePtrOp. The users of this will be the users of
    // the original first UpdFreePtrOp.
    auto fusedUpd = r.create<yul::UpdFreePtrOp>(
        FusedLoc::get(r.getContext(), fusedLocs), totalSize);
    r.replaceAllOpUsesWith(updOps.front(), fusedUpd);

    // The remaining UpdFreePtrOps are generated as add ops and the users are
    // updated accordingly.
    yul::UpdFreePtrOp prevUpdOp = updOps.front();
    Value prevFreePtr = fusedUpd;
    for (yul::UpdFreePtrOp updOp : llvm::drop_begin(updOps)) {
      auto add = r.create<arith::AddIOp>(updOp.getLoc(), prevFreePtr,
                                         prevUpdOp.getSize());
      r.replaceAllOpUsesWith(updOp, add);
      prevFreePtr = add;
      prevUpdOp = updOp;
    }

    // Erase all original UpdFreePtrOps.
    for (yul::UpdFreePtrOp updOp : updOps)
      r.eraseOp(updOp);
  }
};

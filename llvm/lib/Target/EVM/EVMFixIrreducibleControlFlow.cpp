//===- EVMFixIrreducibleControlFlow.cpp - Fix Irreducible Control Flow ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass is inspired by the WebAssemblyFixIrreducibleControlFlow and
// implements a pass that removes irreducible control flow.
// Irreducible control flow means multiple-entry loops, which this pass
// transforms to have a single entry.
//
//   Before — irreducible 2-entry cycle:
//
//          outside
//          /     \
//         v       v
//       Entry0 <--> Entry1
//                    |
//                    v
//                   exit
//
// We rewrite the region so a single dispatch block becomes its only entry.
// Edges into the cycle's entries are split through "route" blocks that
// select the target via a shared label vreg, so the dispatch reads the label
// and forwards execution. Outer routes (for predecessors outside the
// cycle) and inner routes (for predecessors inside it) must be kept
// separate, as a shared route would sit on a back-edge inside the cycle
// while also being reachable from outside, turning that route into a
// second entry and reintroducing irreducibility.
//
//   After — single-entry via dispatch:
//
//                    outside
//                    /     \
//                   v       v
//                outerR0  outerR1
//                    \    /
//                     v  v
//                +-> dispatch
//                |   /     \
//                |  v       v
//                | Entry0  Entry1 -> exit
//                |   |       |
//                |   v       v
//                | innerR1  innerR0
//                |   |       |
//                +---+-------+
//
// After the rewrite, a vreg defined inside the cycle and consumed across
// an entry boundary may have a use that no longer has a reaching def
// along the new dispatch paths, so as a final step we restore liveness.
//
//===----------------------------------------------------------------------===//

#include "EVM.h"
#include "EVMSubtarget.h"
#include "MCTargetDesc/EVMMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "evm-fix-irreducible-control-flow"

namespace {

using BlockVector = SmallVector<MachineBasicBlock *, 4>;
using BlockSet = SmallPtrSet<MachineBasicBlock *, 4>;

// Returns the entries sorted by MBB number, so that downstream processing is
// deterministic.
BlockVector getSortedEntries(const BlockSet &Entries) {
  BlockVector SortedEntries(Entries.begin(), Entries.end());
  llvm::sort(SortedEntries,
             [](const MachineBasicBlock *A, const MachineBasicBlock *B) {
               return A->getNumber() < B->getNumber();
             });
  return SortedEntries;
}

struct ReachabilityNode {
  MachineBasicBlock *MBB = nullptr;
  SmallVector<ReachabilityNode *, 4> Succs;
  unsigned SCCId = std::numeric_limits<unsigned>::max();
};

// Analyzes the SCC (strongly-connected component) structure in a region.
// Ignores branches to blocks outside of the region, and ignores branches to the
// region entry (for the case where the region is the inner part of a loop).
class ReachabilityGraph {
public:
  ReachabilityGraph(MachineBasicBlock *Entry, const BlockSet &Blocks)
      : Entry(Entry), Blocks(Blocks) {
#ifndef NDEBUG
    // The region must have a single entry.
    for (auto *MBB : Blocks)
      if (MBB != Entry)
        for (auto *Pred : MBB->predecessors())
          assert(inRegion(Pred));
#endif
    calculate();
  }

  // Get all blocks that are loop entries.
  const BlockSet &getLoopEntries() const { return LoopEntries; }
  const BlockSet &getLoopEntriesForSCC(unsigned SCCId) const {
    return LoopEntriesBySCC[SCCId];
  }

  unsigned getSCCId(MachineBasicBlock *MBB) const {
    return getNode(MBB)->SCCId;
  }

  friend struct GraphTraits<ReachabilityGraph *>;

private:
  MachineBasicBlock *Entry;
  const BlockSet &Blocks;

  BlockSet LoopEntries;
  SmallVector<BlockSet, 0> LoopEntriesBySCC;

  bool inRegion(MachineBasicBlock *MBB) const { return Blocks.contains(MBB); }

  SmallVector<ReachabilityNode, 0> Nodes;
  DenseMap<MachineBasicBlock *, ReachabilityNode *> MBBToNodeMap;

  ReachabilityNode *getNode(MachineBasicBlock *MBB) const {
    return MBBToNodeMap.at(MBB);
  }

  void calculate();
};

} // end anonymous namespace

namespace llvm {
template <> struct GraphTraits<ReachabilityGraph *> {
  using NodeRef = ReachabilityNode *;
  using ChildIteratorType = SmallVectorImpl<NodeRef>::iterator;

  static NodeRef getEntryNode(ReachabilityGraph *G) {
    return G->getNode(G->Entry);
  }

  static ChildIteratorType child_begin(NodeRef N) { return N->Succs.begin(); }

  static ChildIteratorType child_end(NodeRef N) { return N->Succs.end(); }
};
} // end namespace llvm

namespace {

void ReachabilityGraph::calculate() {
  auto NumBlocks = Blocks.size();
  Nodes.assign(NumBlocks, {});

  MBBToNodeMap.clear();
  MBBToNodeMap.reserve(NumBlocks);

  // Initialize mappings.
  unsigned MBBIdx = 0;
  for (auto *MBB : Blocks) {
    auto &Node = Nodes[MBBIdx++];
    Node.MBB = MBB;
    MBBToNodeMap[MBB] = &Node;
  }

  // Add all relevant direct branches.
  MBBIdx = 0;
  for (auto *MBB : Blocks) {
    auto &Node = Nodes[MBBIdx++];
    for (auto *Succ : MBB->successors())
      if (Succ != Entry && inRegion(Succ))
        Node.Succs.push_back(getNode(Succ));
  }

  unsigned CurrSCCIdx = 0;
  for (const auto &SCC : make_range(scc_begin(this), scc_end(this))) {
    LoopEntriesBySCC.push_back({});
    auto &SCCLoopEntries = LoopEntriesBySCC.back();

    for (auto *Node : SCC) {
      // Make sure nodes are only ever assigned one SCC
      assert(Node->SCCId == std::numeric_limits<unsigned>::max());

      Node->SCCId = CurrSCCIdx;
    }

    bool SelfLoop = false;
    if (SCC.size() == 1) {
      const auto &Node = SCC[0];

      for (auto *Succ : Node->Succs) {
        if (Succ == Node) {
          SelfLoop = true;
          break;
        }
      }
    }

    // Blocks outside any (multi-block) loop will be isolated in their own
    // single-element SCC. Thus blocks that are in a loop are those in
    // multi-element SCCs or are self-looping.
    if (SCC.size() > 1 || SelfLoop) {
      // Find the loop entries - loop body blocks with predecessors outside
      // their SCC
      for (auto *Node : SCC) {
        if (Node->MBB == Entry)
          continue;

        for (auto *Pred : Node->MBB->predecessors()) {
          // This test is accurate despite not having assigned all nodes an SCC
          // yet. We only care if a node has been assigned into this SCC or not.
          if (getSCCId(Pred) != CurrSCCIdx) {
            LoopEntries.insert(Node->MBB);
            SCCLoopEntries.insert(Node->MBB);
          }
        }
      }
    }
    ++CurrSCCIdx;
  }

#ifndef NDEBUG
  // Make sure all nodes have been processed
  for (auto &Node : Nodes)
    assert(Node.SCCId != std::numeric_limits<unsigned>::max());
#endif
}

class EVMFixIrreducibleControlFlow final : public MachineFunctionPass {
public:
  static char ID;

  EVMFixIrreducibleControlFlow() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "EVM Fix Irreducible Control Flow";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  // Recursively fix a single-entry region. Detects multi-entry SCCs in the
  // region and rewrites each via makeSingleEntryLoop; once the region is
  // reducible, descends into each inner loop.
  bool processRegion(MachineBasicBlock *Entry, BlockSet &Blocks,
                     MachineFunction &MF);

  // Rewrites one irreducible cycle to have a single entry. The algorithm:
  //   1. Build a dispatch chain.
  //   2. Create inner and outer route blocks for every entry in the cycle.
  //   3. Redirect each predecessor's entry-targeting edges through its route.
  void makeSingleEntryLoop(const BlockSet &Entries, BlockSet &Blocks,
                           MachineFunction &MF, const ReachabilityGraph &Graph);
};
} // end anonymous namespace

char EVMFixIrreducibleControlFlow::ID = 0;

INITIALIZE_PASS(EVMFixIrreducibleControlFlow, DEBUG_TYPE,
                "Fix irreducible control flow", false, false)

FunctionPass *llvm::createEVMFixIrreducibleControlFlow() {
  return new EVMFixIrreducibleControlFlow();
}

bool EVMFixIrreducibleControlFlow::processRegion(MachineBasicBlock *Entry,
                                                 BlockSet &Blocks,
                                                 MachineFunction &MF) {
  bool Changed = false;

  // Remove irreducibility before processing child loops, which may take
  // multiple iterations.
  while (true) {
    ReachabilityGraph Graph(Entry, Blocks);

    bool FoundIrreducibility = false;
    for (auto *LoopEntry : getSortedEntries(Graph.getLoopEntries())) {
      // Find mutual entries - all entries which can reach this one, and
      // are reached by it (that always includes LoopEntry itself). All mutual
      // entries must be in the same SCC, so if we have more than one, then we
      // have irreducible control flow.
      //
      // (Note that we need to sort the entries here, as otherwise the order can
      // matter: being mutual is a symmetric relationship, and each set of
      // mutuals will be handled properly no matter which we see first. However,
      // there can be multiple disjoint sets of mutuals, and which we process
      // first changes the output.)
      //
      // Note that irreducibility may involve inner loops, e.g. imagine A
      // starts one loop, and it has B inside it which starts an inner loop.
      // If we add a branch from all the way on the outside to B, then in a
      // sense B is no longer an "inner" loop, semantically speaking. We will
      // fix that irreducibility by adding a block that dispatches to either
      // either A or B, so B will no longer be an inner loop in our output.
      // (A fancier approach might try to keep it as such.)
      //
      // Note that we still need to recurse into inner loops later, to handle
      // the case where the irreducibility is entirely nested - we would not
      // be able to identify that at this point, since the enclosing loop is
      // a group of blocks all of whom can reach each other. (We'll see the
      // irreducibility after removing branches to the top of that enclosing
      // loop.)
      const auto &MutualLoopEntries =
          Graph.getLoopEntriesForSCC(Graph.getSCCId(LoopEntry));
      if (MutualLoopEntries.size() > 1) {
        makeSingleEntryLoop(MutualLoopEntries, Blocks, MF, Graph);
        FoundIrreducibility = true;
        Changed = true;
        break;
      }
    }

    // Only go on to actually process the inner loops when we are done
    // removing irreducible control flow and changing the graph. Modifying
    // the graph as we go is possible, and that might let us avoid looking at
    // the already-fixed loops again if we are careful, but all that is
    // complex and bug-prone. Since irreducible loops are rare, just starting
    // another iteration is best.
    if (FoundIrreducibility)
      continue;

    for (auto *LoopEntry : Graph.getLoopEntries()) {
      BlockSet InnerBlocks;

      auto EntrySCCId = Graph.getSCCId(LoopEntry);
      for (auto *Block : Blocks)
        if (EntrySCCId == Graph.getSCCId(Block))
          InnerBlocks.insert(Block);

      // Each of these calls to processRegion may change the graph, but are
      // guaranteed not to interfere with each other. The only changes we make
      // to the graph are to add blocks on the way to a loop entry. As the
      // loops are disjoint, that means we may only alter branches that exit
      // another loop, which are ignored when recursing into that other loop
      // anyhow.
      if (processRegion(LoopEntry, InnerBlocks, MF))
        Changed = true;
    }
    return Changed;
  }
}

void EVMFixIrreducibleControlFlow::makeSingleEntryLoop(
    const BlockSet &Entries, BlockSet &Blocks, MachineFunction &MF,
    const ReachabilityGraph &Graph) {
  const unsigned NumEntries = Entries.size();
  assert(NumEntries >= 2 && "irreducible cycle must have >= 2 entries");

  // Sort the entries by MBB number to get a stable order for downstream
  // processing.
  BlockVector SortedEntries = getSortedEntries(Entries);
  LLVM_DEBUG({
    dbgs() << "Fixing irreducible CFG with entries:\n  ";
    bool NeedComma = false;
    for (auto *Entry : SortedEntries) {
      if (NeedComma)
        dbgs() << ", ";
      Entry->printName(dbgs());
      NeedComma = true;
    }
    dbgs() << "\n";
  });

  DenseMap<const MachineBasicBlock *, unsigned> EntryIdx;
  SetVector<MachineBasicBlock *> AllPreds;

  // Cache entry indices and all predecessors of entries, so we can use them in
  // the processing below.
  for (auto [I, Entry] : llvm::enumerate(SortedEntries)) {
    EntryIdx[Entry] = I;
    for (MachineBasicBlock *Pred : Entry->predecessors())
      AllPreds.insert(Pred);
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  LLVMContext &Ctx = MF.getFunction().getContext();

  // The shared label vreg. Every route block writes its entry's index into
  // it, and every dispatch block reads it to decide which entry to jump to.
  Register Label = MRI.createVirtualRegister(&EVM::GPRRegClass);

  SmallVector<MachineBasicBlock *> Dispatches;
  Dispatches.reserve(NumEntries - 1);

  // 1. Build a dispatch chain. We need NumEntries - 1 dispatch blocks and each
  // checks one non-default entry and jumps to it on a match, otherwise falling
  // through to the next dispatch block. The last entry is the default case, so
  // the final dispatch block ends with an unconditional jump to it.
  // Dispatch blocks will look like this:
  //   %bb.dispatch0:
  //     JUMP_UNLESS %bb.entry0, %label
  //     // fallthrough
  //   %bb.dispatch1:
  //     %cond = EQ %label, 1
  //     JUMPI %bb.entry1, %cond
  //     // fallthrough
  //   ...
  //   %bb.dispatchN-1:
  //     %cond = EQ %label, N-1
  //     JUMPI %bb.entryN-1, %cond
  //     JUMP %bb.entryN
  for (unsigned I = 0; I + 1 < NumEntries; ++I) {
    auto *Entry = SortedEntries[I];
    auto *D = MF.CreateMachineBasicBlock();
    MF.insert(MF.end(), D);
    Blocks.insert(D);
    Dispatches.push_back(D);

    if (I == 0) {
      // For zero label we can just simply do:
      //   JUMP_UNLESS %bb.entry, %label
      BuildMI(D, DebugLoc(), TII.get(EVM::JUMP_UNLESS))
          .addMBB(Entry)
          .addReg(Label);
    } else {
      // For the rest, emit:
      //   %cond = EQ %label, %idx
      //   JUMPI %bb.entry, %cond
      Register Idx = MRI.createVirtualRegister(&EVM::GPRRegClass);
      BuildMI(D, DebugLoc(), TII.get(EVM::CONST_I256), Idx)
          .addCImm(ConstantInt::get(Ctx, APInt(256, I)));

      Register Cond = MRI.createVirtualRegister(&EVM::GPRRegClass);
      BuildMI(D, DebugLoc(), TII.get(EVM::EQ), Cond).addReg(Label).addReg(Idx);

      BuildMI(D, DebugLoc(), TII.get(EVM::JUMPI)).addMBB(Entry).addReg(Cond);
    }
    D->addSuccessor(Entry);

    // Update the previous dispatch block to fallthrough here.
    if (I > 0)
      Dispatches[I - 1]->addSuccessor(D);
  }

  // For the default entry, we just emit an unconditional JUMP at the end of
  // the last dispatch block.
  auto *LastDispatch = Dispatches.back();
  BuildMI(LastDispatch, DebugLoc(), TII.get(EVM::JUMP))
      .addMBB(SortedEntries.back());
  LastDispatch->addSuccessor(SortedEntries.back());

  DenseSet<MachineBasicBlock *> InLoopPreds;

  // Cache all predecessors of entries that are inside the cycle, so we
  // can use it in the processing below.
  for (auto *Pred : AllPreds) {
    auto PredSCCId = Graph.getSCCId(Pred);
    for (auto *Succ : Pred->successors()) {
      if (EntryIdx.contains(Succ) && Graph.getSCCId(Succ) == PredSCCId) {
        InLoopPreds.insert(Pred);
        break;
      }
    }
  }

  using RouteKey = PointerIntPair<MachineBasicBlock *, 1, bool>;
  DenseMap<RouteKey, MachineBasicBlock *> EntryToLayoutPred;

  // Find layout predecessors for entries, if any.
  for (auto *Pred : AllPreds) {
    const bool PredInLoop = InLoopPreds.contains(Pred);
    for (auto *Succ : Pred->successors())
      if (EntryIdx.contains(Succ) && Pred->isLayoutSuccessor(Succ))
        EntryToLayoutPred[{Succ, PredInLoop}] = Pred;
  }

  DenseMap<RouteKey, MachineBasicBlock *> RouteMap;
  auto *Dispatch0 = Dispatches.front();

  // 2. Generate at most two routing blocks per entry: one for predecessors
  // outside the loop and one for predecessors inside the loop.
  // Route blocks will contain:
  //   %label = CONST_I256 %idx
  //   JUMP %bb.dispatch0
  for (auto *Pred : AllPreds) {
    const bool PredInLoop = InLoopPreds.contains(Pred);
    for (auto *Succ : Pred->successors()) {
      RouteKey Key{Succ, PredInLoop};
      auto IdxIt = EntryIdx.find(Succ);

      // Create a route if the edge targets an entry, and we haven't already
      // created a route.
      if (IdxIt == EntryIdx.end() || RouteMap.contains(Key))
        continue;

      // If an entry has a layout predecessor, create the route while processing
      // that predecessor so it can be placed right before the entry and stay on
      // the fall-through path, avoiding an extra jump. Skip the other
      // predecessors so the shared route is created for the layout predecessor.
      if (auto *LayoutPred = EntryToLayoutPred.lookup(Key))
        if (LayoutPred != Pred)
          continue;

      auto *Route = MF.CreateMachineBasicBlock();
      MF.insert(Pred->isLayoutSuccessor(Succ) ? MachineFunction::iterator(Succ)
                                              : MF.end(),
                Route);

      // The route is now part of the region, so add it to Blocks. Later
      // processRegion iterations rebuild the reachability/SCC analysis over
      // Blocks, and they must account for the route to analyze the region
      // correctly.
      Blocks.insert(Route);
      BuildMI(Route, DebugLoc(), TII.get(EVM::CONST_I256), Label)
          .addCImm(ConstantInt::get(Ctx, APInt(256, IdxIt->second)));
      BuildMI(Route, DebugLoc(), TII.get(EVM::JUMP)).addMBB(Dispatch0);
      Route->addSuccessor(Dispatch0);
      RouteMap[Key] = Route;
    }
  }

  // 3. Update all predecessors to target their route instead of the entry
  // directly.
  for (auto *Pred : AllPreds) {
    const bool PredInLoop = InLoopPreds.contains(Pred);
    SmallVector<MachineBasicBlock *> OrigEntrySuccs;

    // Cache the original entry successors before mutating the successor list.
    for (auto *Succ : Pred->successors())
      if (EntryIdx.contains(Succ))
        OrigEntrySuccs.push_back(Succ);

    // Redirect explicit branches.
    for (auto &Term : Pred->terminators())
      for (auto &Op : Term.explicit_uses())
        if (Op.isMBB() && EntryIdx.contains(Op.getMBB()))
          Op.setMBB(RouteMap[{Op.getMBB(), PredInLoop}]);

    // Sync the successor list with the rewritten terminators.
    for (auto *OrigSucc : OrigEntrySuccs)
      Pred->replaceSuccessor(OrigSucc, RouteMap[{OrigSucc, PredInLoop}]);
  }
}

// Add a register definition with IMPLICIT_DEFs for every register to cover for
// register uses that don't have defs in every possible path.
// FIXME: Find a better approach than this.
static void addImplicitDefs(MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  auto &EntryBB = MF.front();

  auto isArgumentOrPushDeployAddress = [](const MachineInstr &MI) {
    return MI.getOpcode() == EVM::ARGUMENT ||
           MI.getOpcode() == EVM::PUSHDEPLOYADDRESS;
  };

  // Skip argument and pushdeployaddress instructions, as they have to be at the
  // top of the entry block.
  MachineBasicBlock::iterator InsertPt = EntryBB.begin();
  while (InsertPt != EntryBB.end() && isArgumentOrPushDeployAddress(*InsertPt))
    ++InsertPt;

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.use_nodbg_empty(Reg) ||
        llvm::any_of(MRI.def_instructions(Reg), isArgumentOrPushDeployAddress))
      continue;
    BuildMI(EntryBB, InsertPt, DebugLoc(), TII.get(EVM::IMPLICIT_DEF), Reg);
  }
}

bool EVMFixIrreducibleControlFlow::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG({
    dbgs() << "********** Fix Irreducible Control Flow **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  // Seed the recursion with the whole function as a single-entry region.
  BlockSet AllBlocks;
  for (auto &MBB : MF)
    AllBlocks.insert(&MBB);

  if (LLVM_UNLIKELY(processRegion(&MF.front(), AllBlocks, MF))) {
    // We rewrote part of the function; recompute relevant things.
    MF.RenumberBlocks();
    // Now we've inserted dispatch blocks, some register uses can have incoming
    // paths without a def. For example, before this pass register %a was
    // defined in BB1 and used in BB2, and there was only one path from BB1 and
    // BB2. But if this pass inserts a dispatch block having multiple
    // predecessors between the two BBs, now there are paths to BB2 without
    // visiting BB1, and %a's use in BB2 is not dominated by its def. Adding
    // IMPLICIT_DEFs to all regs is one simple way to fix it.
    addImplicitDefs(MF);
    return true;
  }
  return false;
}

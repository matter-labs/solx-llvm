// RUN: mlir-opt %s --convert-sol-to-yul | FileCheck %s

#Default = #sol<RevertStrings Default>
#Osaka = #sol<EvmVersion Osaka>
module attributes {llvm.data_layout = "E-p:256:256-i256:256:256-S256-a:256:256",
                   llvm.target_triple = "evm-unknown-unknown",
                   sol.evm_version = #Osaka,
                   sol.revert_strings = #Default} {

  // CHECK-LABEL: yul.func @band : (i256, i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256, %arg1: i256):
  // CHECK-NEXT:    %0 = yul.and %arg0, %arg1
  // CHECK-NEXT:    yul.func_return %0 : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  sol.func @band(%arg0: !sol.byte, %arg1: !sol.byte) -> !sol.byte {
    %0 = sol.and %arg0, %arg1 : !sol.byte
    sol.return %0 : !sol.byte
  }

  // CHECK-LABEL: yul.func @bor : (i256, i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256, %arg1: i256):
  // CHECK-NEXT:    %0 = yul.or %arg0, %arg1
  // CHECK-NEXT:    yul.func_return %0 : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  sol.func @bor(%arg0: !sol.byte, %arg1: !sol.byte) -> !sol.byte {
    %0 = sol.or %arg0, %arg1 : !sol.byte
    sol.return %0 : !sol.byte
  }

  // CHECK-LABEL: yul.func @bxor : (i256, i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256, %arg1: i256):
  // CHECK-NEXT:    %0 = yul.xor %arg0, %arg1
  // CHECK-NEXT:    yul.func_return %0 : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  sol.func @bxor(%arg0: !sol.byte, %arg1: !sol.byte) -> !sol.byte {
    %0 = sol.xor %arg0, %arg1 : !sol.byte
    sol.return %0 : !sol.byte
  }

  // CHECK-LABEL: yul.func @bnot : (i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256):
  // CHECK-NEXT:    %0 = yul.not %arg0
  // CHECK-NEXT:    yul.func_return %0 : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  sol.func @bnot(%arg0: !sol.byte) -> !sol.byte {
    %0 = sol.not %arg0 : !sol.byte
    sol.return %0 : !sol.byte
  }

  // CleanedOperandsLowering applies an AND mask (top 8 bits = bytes1 mask)
  // to the lhs before the shift, and an AND 255 mask to the rhs (ui8 cleanup).
  // CHECK-LABEL: yul.func @sshl : (i256, i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256, %arg1: i256):
  // CHECK-NEXT:    %[[BMASK:.*]] = yul.constant -452312848583266388373324160190187140051835877600158453279131187530910662656
  // CHECK-NEXT:    %[[LHS:.*]] = yul.and %arg0, %[[BMASK]]
  // CHECK-NEXT:    %[[UMASK:.*]] = yul.constant 255
  // CHECK-NEXT:    %[[RHS:.*]] = yul.and %arg1, %[[UMASK]]
  // CHECK-NEXT:    %[[RES:.*]] = yul.shl %[[RHS]], %[[LHS]]
  // CHECK-NEXT:    yul.func_return %[[RES]] : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  sol.func @sshl(%arg0: !sol.byte, %arg1: ui8) -> !sol.byte {
    %0 = sol.shl %arg0, %arg1 : !sol.byte, ui8
    sol.return %0 : !sol.byte
  }

  // sol.shr on !sol.byte must lower to yul.shr (logical), not yul.sar
  // (arithmetic), because ByteType is unsigned.
  // CHECK-LABEL: yul.func @sshr : (i256, i256) -> i256 {
  // CHECK-NEXT:  ^bb0(%arg0: i256, %arg1: i256):
  // CHECK-NEXT:    %[[BMASK:.*]] = yul.constant -452312848583266388373324160190187140051835877600158453279131187530910662656
  // CHECK-NEXT:    %[[LHS:.*]] = yul.and %arg0, %[[BMASK]]
  // CHECK-NEXT:    %[[UMASK:.*]] = yul.constant 255
  // CHECK-NEXT:    %[[RHS:.*]] = yul.and %arg1, %[[UMASK]]
  // CHECK-NEXT:    %[[RES:.*]] = yul.shr %[[RHS]], %[[LHS]]
  // CHECK-NEXT:    yul.func_return %[[RES]] : i256
  // CHECK-NEXT:  } {llvm.linkage = #llvm.linkage<private>}
  // CHECK-NOT:   yul.sar
  sol.func @sshr(%arg0: !sol.byte, %arg1: ui8) -> !sol.byte {
    %0 = sol.shr %arg0, %arg1 : !sol.byte, ui8
    sol.return %0 : !sol.byte
  }
}

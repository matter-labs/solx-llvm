; RUN: not opt -passes=evm-verifier -disable-output %s 2>&1 | FileCheck %s

target datalayout = "E-p:256:256-i256:256:256-S256-a:256:256"
target triple = "evm"

declare void @llvm.memcpy.p1.p1.i256(ptr addrspace(1), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p1.p2.i256(ptr addrspace(1), ptr addrspace(2), i256, i1 immarg)
declare void @llvm.memcpy.p1.p3.i256(ptr addrspace(1), ptr addrspace(3), i256, i1 immarg)
declare void @llvm.memcpy.p1.p4.i256(ptr addrspace(1), ptr addrspace(4), i256, i1 immarg)
declare void @llvm.memmove.p1.p1.i256(ptr addrspace(1), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memmove.p1.p2.i256(ptr addrspace(1), ptr addrspace(2), i256, i1 immarg)
declare void @llvm.memmove.p1.p3.i256(ptr addrspace(1), ptr addrspace(3), i256, i1 immarg)
declare void @llvm.memmove.p1.p4.i256(ptr addrspace(1), ptr addrspace(4), i256, i1 immarg)

declare void @llvm.memcpy.p0.p1.i256(ptr addrspace(0), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p2.p1.i256(ptr addrspace(2), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p3.p1.i256(ptr addrspace(3), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p4.p1.i256(ptr addrspace(4), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p5.p1.i256(ptr addrspace(5), ptr addrspace(1), i256, i1 immarg)
declare void @llvm.memcpy.p6.p1.i256(ptr addrspace(6), ptr addrspace(1), i256, i1 immarg)

declare void @llvm.memmove.p1.p0.i256(ptr addrspace(1), ptr addrspace(0), i256, i1 immarg)
declare void @llvm.memmove.p1.p5.i256(ptr addrspace(1), ptr addrspace(5), i256, i1 immarg)
declare void @llvm.memmove.p1.p6.i256(ptr addrspace(1), ptr addrspace(6), i256, i1 immarg)

declare void @llvm.memset.p1.i256(ptr addrspace(1), i8, i256, i1 immarg)
declare void @llvm.memset.p5.i256(ptr addrspace(5), i8, i256, i1 immarg)

; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p0.p1.i256
; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p2.p1.i256
; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p3.p1.i256
; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p4.p1.i256
; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p5.p1.i256
; CHECK: EVM memcpy/memmove must write to heap memory
; CHECK-NEXT: call void @llvm.memcpy.p6.p1.i256
; CHECK: EVM memcpy/memmove must read from heap, calldata, returndata, or code memory
; CHECK-NEXT: call void @llvm.memmove.p1.p0.i256
; CHECK: EVM memcpy/memmove must read from heap, calldata, returndata, or code memory
; CHECK-NEXT: call void @llvm.memmove.p1.p5.i256
; CHECK: EVM memcpy/memmove must read from heap, calldata, returndata, or code memory
; CHECK-NEXT: call void @llvm.memmove.p1.p6.i256
; CHECK: EVM does not support generic memset intrinsics
; CHECK-NEXT: call void @llvm.memset.p1.i256
; CHECK: EVM does not support generic memset intrinsics
; CHECK-NEXT: call void @llvm.memset.p5.i256
define void @invalid(ptr addrspace(0) %stack,
                     ptr addrspace(1) %heap,
                     ptr addrspace(2) %calldata,
                     ptr addrspace(3) %returndata,
                     ptr addrspace(4) %code,
                     ptr addrspace(5) %storage,
                     ptr addrspace(6) %tstorage,
                     i256 %len) {
  call void @llvm.memcpy.p0.p1.i256(ptr addrspace(0) %stack, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p2.p1.i256(ptr addrspace(2) %calldata, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p3.p1.i256(ptr addrspace(3) %returndata, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p4.p1.i256(ptr addrspace(4) %code, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p5.p1.i256(ptr addrspace(5) %storage, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p6.p1.i256(ptr addrspace(6) %tstorage, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memmove.p1.p0.i256(ptr addrspace(1) %heap, ptr addrspace(0) %stack, i256 %len, i1 false)
  call void @llvm.memmove.p1.p5.i256(ptr addrspace(1) %heap, ptr addrspace(5) %storage, i256 %len, i1 false)
  call void @llvm.memmove.p1.p6.i256(ptr addrspace(1) %heap, ptr addrspace(6) %tstorage, i256 %len, i1 false)
  call void @llvm.memset.p1.i256(ptr addrspace(1) %heap, i8 0, i256 %len, i1 false)
  call void @llvm.memset.p5.i256(ptr addrspace(5) %storage, i8 0, i256 %len, i1 false)
  ret void
}

; CHECK-NOT: EVM memcpy/memmove must write to heap memory
; CHECK-NOT: EVM memcpy/memmove must read from heap, calldata, returndata, or code memory
define void @valid(ptr addrspace(1) %heap,
                   ptr addrspace(2) %calldata,
                   ptr addrspace(3) %returndata,
                   ptr addrspace(4) %code,
                   i256 %len) {
  call void @llvm.memcpy.p1.p1.i256(ptr addrspace(1) %heap, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memcpy.p1.p2.i256(ptr addrspace(1) %heap, ptr addrspace(2) %calldata, i256 %len, i1 false)
  call void @llvm.memcpy.p1.p3.i256(ptr addrspace(1) %heap, ptr addrspace(3) %returndata, i256 %len, i1 false)
  call void @llvm.memcpy.p1.p4.i256(ptr addrspace(1) %heap, ptr addrspace(4) %code, i256 %len, i1 false)
  call void @llvm.memmove.p1.p1.i256(ptr addrspace(1) %heap, ptr addrspace(1) %heap, i256 %len, i1 false)
  call void @llvm.memmove.p1.p2.i256(ptr addrspace(1) %heap, ptr addrspace(2) %calldata, i256 %len, i1 false)
  call void @llvm.memmove.p1.p3.i256(ptr addrspace(1) %heap, ptr addrspace(3) %returndata, i256 %len, i1 false)
  call void @llvm.memmove.p1.p4.i256(ptr addrspace(1) %heap, ptr addrspace(4) %code, i256 %len, i1 false)
  ret void
}

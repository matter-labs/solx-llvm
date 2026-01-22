; RUN: llc -O2 -filetype=obj -split-dwarf-output=%t.dwo < %s | llvm-readobj --sections - | FileCheck %s --check-prefix=OBJ
; RUN: llvm-readobj --sections %t.dwo | FileCheck %s --check-prefix=DWO
; RUN: not --crash llc -O2 -filetype=obj -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo < %s 2>&1 | FileCheck %s -check-prefix=SPLIT-NOT-SUPPORTED

target datalayout = "E-p:256:256-i256:256:256-S256-a:256:256"
target triple = "evm"

; OBJ-NOT: Name: .debug
; DWO: Name: .debug
; SPLIT-NOT-SUPPORTED: LLVM ERROR: EVM does not support split dwarf

define i256 @foo(i256 %0, i256 %1) !dbg !4 {
  %add1 = add nsw i256 %0, 5, !dbg !7
  %add2 = add nsw i256 %1, 3, !dbg !8
  %mul = mul nsw i256 %add2, %add1, !dbg !9
  %add3 = add nsw i256 %add2, %add1, !dbg !10
  %add4 = add nsw i256 %add3, %mul, !dbg !11
  ret i256 %add4, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Assembly, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "Test.sol", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 2, column: 7, scope: !4)
!8 = !DILocation(line: 3, column: 7, scope: !4)
!9 = !DILocation(line: 4, column: 17, scope: !4)
!10 = !DILocation(line: 5, column: 17, scope: !4)
!11 = !DILocation(line: 6, column: 16, scope: !4)
!12 = !DILocation(line: 6, column: 5, scope: !4)

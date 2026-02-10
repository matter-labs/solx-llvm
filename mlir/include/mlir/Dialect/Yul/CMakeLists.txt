# Create MLIRYulOpsIncGen.
add_mlir_dialect(YulOps yul)

# Generate sources from YulOps.td.
set(LLVM_TARGET_DEFINITIONS YulOps.td)

# Generate documentation.
add_mlir_doc(YulDialect YulDialect Yul/ -gen-dialect-doc)
add_mlir_doc(YulOps YulOps Yul/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Transforms/Passes.td)
mlir_tablegen(Transforms/Passes.h.inc -gen-pass-decls -name Yul)
mlir_tablegen(Transforms/Passes.capi.h.inc -gen-pass-capi-header --prefix Yul)
mlir_tablegen(Transforms/Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Yul)
add_public_tablegen_target(MLIRYulPassIncGen)

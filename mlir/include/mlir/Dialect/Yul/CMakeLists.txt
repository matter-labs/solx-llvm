# Create MLIRYulOpsIncGen.
add_mlir_dialect(YulOps yul)

# Generate sources from YulOps.td.
set(LLVM_TARGET_DEFINITIONS YulOps.td)

# Generate documentation.
add_mlir_doc(YulDialect YulDialect Yul/ -gen-dialect-doc)
add_mlir_doc(YulOps YulOps Yul/ -gen-op-doc)

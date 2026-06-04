add_mlir_dialect_library(MLIRSolDialect
  SolBase.cpp
  SolOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Sol

  DEPENDS
  MLIRSolOpsIncGen
  MLIRSolExtraIncGen
  MLIRYulOpsIncGen
  MLIRYulExtraIncGen
  MLIRYulInterfacesIncGen

  LINK_LIBS PUBLIC
  # Bridge ops (sol.ptr_cast, sol.selector, ...) produce !yul.ptr.
  MLIRYulDialect
  )

add_subdirectory(Transforms)

add_mlir_dialect_library(MLIRSolDialect
  SolBase.cpp
  SolOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Sol

  DEPENDS
  MLIRSolOpsIncGen
  MLIRSolExtraIncGen

  LINK_LIBS PUBLIC
  )

add_subdirectory(Transforms)

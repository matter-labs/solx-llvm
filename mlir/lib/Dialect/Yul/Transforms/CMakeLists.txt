add_mlir_dialect_library(MLIRYulTransforms
  FuseFreePtr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Yul/Transforms

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRYulDialect

  LINK_COMPONENTS
  Support
  )

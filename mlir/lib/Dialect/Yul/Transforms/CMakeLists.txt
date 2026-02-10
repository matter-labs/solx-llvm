add_mlir_dialect_library(MLIRYulTransforms
  YulFuseFreePtr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Yul/Transforms

  DEPENDS
  MLIRYulPassIncGen

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

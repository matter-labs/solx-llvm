add_mlir_dialect_library(MLIRSolTransforms
  Immutables.cpp
  LoopInvariantCodeMotion.cpp
  ModifierOpLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Sol/Transforms

  LINK_LIBS PUBLIC
  MLIRSolToStandard
  MLIRSolDialect
  MLIRYulDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils

  LINK_COMPONENTS
  Support
  )

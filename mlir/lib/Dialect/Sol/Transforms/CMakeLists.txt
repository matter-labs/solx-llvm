add_mlir_dialect_library(MLIRSolTransforms
  SolImmutables.cpp
  SolLoopInvariantCodeMotion.cpp
  SolModifierOpLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Sol/Transforms

  DEPENDS
  MLIRSolPassIncGen

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

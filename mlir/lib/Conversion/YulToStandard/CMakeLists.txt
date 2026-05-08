add_mlir_conversion_library(MLIRYulToStandard
  YulToStandard.cpp
  YulToStandardPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/YulToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSolToYul
  MLIRYulDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils

  LINK_COMPONENTS
  Support
  )

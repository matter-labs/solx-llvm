add_mlir_conversion_library(MLIRSolToStandard
  EVMUtil.cpp
  SolToStandardPass.cpp
  SolToYul.cpp
  TypeConverter.cpp
  Util.cpp
  YulToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SolToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSolDialect
  MLIRYulDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils

  LINK_COMPONENTS
  Support
  )

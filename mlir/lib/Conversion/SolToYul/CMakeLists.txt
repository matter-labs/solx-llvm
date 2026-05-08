add_mlir_conversion_library(MLIRSolToYul
  EVMUtil.cpp
  SolToYul.cpp
  SolToYulPass.cpp
  TypeConverter.cpp
  Util.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SolToYul

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSolDialect
  MLIRYulDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils

  LINK_COMPONENTS
  Support
  )

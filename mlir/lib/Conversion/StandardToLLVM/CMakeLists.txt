add_mlir_conversion_library(MLIRStandardToLLVM
  StandardToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToLLVM

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSCFToControlFlow
  MLIRPass
  MLIRTransforms

  LINK_COMPONENTS
  Support
  )

//===- TypeConverter.cpp - Sol dialect type converter ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EVM specific type converter of the sol dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SolToStandard/SolToYul.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Sol/Sol.h"

using namespace mlir;

evm::SolTypeConverter::SolTypeConverter() {
  // Default case
  addConversion([](Type ty) { return ty; });

  // Integer type
  addConversion([&](IntegerType ty) -> Type {
    // Map to signless variant.

    if (ty.isSignless())
      return ty;
    return IntegerType::get(ty.getContext(), ty.getWidth(),
                            IntegerType::Signless);
  });

  // Bytes type
  addConversion([&](sol::BytesType ty) -> Type {
    return IntegerType::get(ty.getContext(), /*width=*/256,
                            IntegerType::Signless);
  });

  // Enum type
  addConversion([&](sol::EnumType ty) -> Type {
    // Unlike integer types, legalizing to i256 avoids unnecessary zext/trunc
    // ops.
    return IntegerType::get(ty.getContext(), 256, IntegerType::Signless);
  });

  // Function ref type
  addConversion([&](sol::FuncRefType ty) -> Type {
    // Maps to the sol.func id used in the dispatch table generated during the
    // indirect call lowering.
    return IntegerType::get(ty.getContext(), 256, IntegerType::Signless);
  });

  // Function type
  addConversion([&](FunctionType ty) -> Type {
    SmallVector<Type> convertedInpTys, convertedResTys;
    if (failed(convertTypes(ty.getInputs(), convertedInpTys)))
      llvm_unreachable("Invalid type");
    if (failed(convertTypes(ty.getResults(), convertedResTys)))
      llvm_unreachable("Invalid type");

    return FunctionType::get(ty.getContext(), convertedInpTys, convertedResTys);
  });

  // Array type
  addConversion([&](sol::ArrayType ty) -> Type {
    auto i256Ty = IntegerType::get(ty.getContext(), 256,
                                   IntegerType::SignednessSemantics::Signless);
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Stack: {
      Type eltTy = convertType(ty.getEltType());
      return LLVM::LLVMArrayType::get(eltTy, ty.getSize());
    }

    // Map to the 256 bit address in calldata/memory.
    // FIXME: Generate fat pointer for calldata!
    case sol::DataLocation::CallData:
      if (ty.isDynSized())
        return LLVM::LLVMStructType::getLiteral(ty.getContext(),
                                                {i256Ty, i256Ty});
      return i256Ty;

    case sol::DataLocation::Memory:
    // Map to the 256 bit slot offset.
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // String type
  addConversion([&](sol::StringType ty) -> Type {
    auto i256Ty = IntegerType::get(ty.getContext(), 256,
                                   IntegerType::SignednessSemantics::Signless);
    switch (ty.getDataLocation()) {
    // Map to the 256 bit address in calldata/memory.
    case sol::DataLocation::CallData:
      return LLVM::LLVMStructType::getLiteral(ty.getContext(),
                                              {i256Ty, i256Ty});
    case sol::DataLocation::Memory:
    // Map to the 256 bit slot offset.
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Mapping type
  addConversion([&](sol::MappingType ty) -> Type {
    // Map to the 256 bit slot offset.
    return IntegerType::get(ty.getContext(), 256,
                            IntegerType::SignednessSemantics::Signless);
  });

  // Struct type
  addConversion([&](sol::StructType ty) -> Type {
    switch (ty.getDataLocation()) {
    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory:
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);
    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Pointer type
  addConversion([&](sol::PointerType ty) -> Type {
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Stack: {
      return LLVM::LLVMPointerType::get(ty.getContext());
    }

    // Map to the 256 bit address in calldata/memory/immutable.
    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory:
    case sol::DataLocation::Immutable:
    // Map to the 256 bit slot offset.
    //
    // TODO: Can we get all storage types to be 32 byte aligned? If so, we can
    // avoid the byte offset. Otherwise we should consider the
    // OneToNTypeConversion to map the pointer to the slot + byte offset pair.
    case sol::DataLocation::Storage:
      return IntegerType::get(ty.getContext(), 256,
                              IntegerType::SignednessSemantics::Signless);
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  addSourceMaterialization(
      [](OpBuilder &b, Type resTy, ValueRange ins, Location loc) -> Value {
        assert(ins.size() == 1);
        return b.create<sol::ConvCastOp>(loc, resTy, ins);
      });
  addTargetMaterialization(
      [](OpBuilder &b, Type resTy, ValueRange ins, Location loc) -> Value {
        assert(ins.size() == 1);
        return b.create<sol::ConvCastOp>(loc, resTy, ins);
      });
}

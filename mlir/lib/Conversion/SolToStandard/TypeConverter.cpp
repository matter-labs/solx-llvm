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

#include "mlir/Conversion/SolToStandard/EVMUtil.h"
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

  // Address type
  addConversion([&](sol::AddressType ty) -> Type {
    return IntegerType::get(ty.getContext(), 256, IntegerType::Signless);
  });

  // Contract type
  addConversion([&](sol::ContractType ty) -> Type {
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

    // Map to fat pointer {addr, size}.
    case sol::DataLocation::CallData:
      if (ty.isDynSized())
        return LLVM::LLVMStructType::getLiteral(ty.getContext(),
                                                {i256Ty, i256Ty});
      return i256Ty;

    // Map to memory address.
    case sol::DataLocation::Memory:
      return i256Ty;

    // Map to storage/transient slot.
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient:
      return i256Ty;

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
    // Map to fat pointer {addr, size}.
    case sol::DataLocation::CallData:
      return LLVM::LLVMStructType::getLiteral(ty.getContext(),
                                              {i256Ty, i256Ty});

    // Map to memory address.
    case sol::DataLocation::Memory:
      return i256Ty;

    // Map to storage/transient slot.
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient:
      return i256Ty;

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Mapping type
  addConversion([&](sol::MappingType ty) -> Type {
    // Map to storage slot.
    return IntegerType::get(ty.getContext(), 256,
                            IntegerType::SignednessSemantics::Signless);
  });

  // Struct type
  addConversion([&](sol::StructType ty) -> Type {
    auto i256Ty = IntegerType::get(ty.getContext(), 256,
                                   IntegerType::SignednessSemantics::Signless);
    switch (ty.getDataLocation()) {
    // Map to calldata/memory address.
    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory:
      return i256Ty;

    // Map to storage/transient slot.
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient:
      return i256Ty;

    default:
      break;
    }

    llvm_unreachable("Unimplemented type conversion");
  });

  // Pointer type
  addConversion([&](sol::PointerType ty) -> Type {
    auto i256Ty = IntegerType::get(ty.getContext(), 256,
                                   IntegerType::SignednessSemantics::Signless);
    switch (ty.getDataLocation()) {
    case sol::DataLocation::Stack:
      return LLVM::LLVMPointerType::get(ty.getContext());

    // Map to calldata/memory address.
    case sol::DataLocation::CallData:
    case sol::DataLocation::Memory:
    case sol::DataLocation::Immutable:
      return i256Ty;

    // Map to fat pointer {slot, offset} for packable types, just slot
    // otherwise.
    case sol::DataLocation::Storage:
    case sol::DataLocation::Transient:
      if (evm::canBePacked(ty.getPointeeType()))
        return LLVM::LLVMStructType::getLiteral(ty.getContext(),
                                                {i256Ty, i256Ty});
      return i256Ty;
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

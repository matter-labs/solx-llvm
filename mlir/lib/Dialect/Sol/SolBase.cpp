//===- SolDialect.cpp - MLIR Dialect for Solidity implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Solidity dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Sol/Sol.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Sol/SolInterfaces.cpp.inc"
#include "mlir/Dialect/Sol/SolOpsDialect.cpp.inc"
#include "mlir/Dialect/Sol/SolOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::sol;

namespace {

struct SolOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto contrKindAttr = dyn_cast<ContractKindAttr>(attr)) {
      os << stringifyContractKind(contrKindAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto stateMutAttr = dyn_cast<StateMutabilityAttr>(attr)) {
      os << stringifyStateMutability(stateMutAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto fnKindAttr = dyn_cast<FunctionKindAttr>(attr)) {
      os << stringifyFunctionKind(fnKindAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto evmVersionAttr = dyn_cast<EvmVersionAttr>(attr)) {
      os << stringifyEvmVersion(evmVersionAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    if (auto revertStringsAttr = dyn_cast<RevertStringsAttr>(attr)) {
      os << stringifyRevertStrings(revertStringsAttr.getValue());
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

} // namespace

void SolDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Sol/SolOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Sol/SolOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Sol/SolOpsAttributes.cpp.inc"
      >();

  addInterfaces<SolOpAsmDialectInterface>();
}

Operation *SolDialect::materializeConstant(OpBuilder &builder, Attribute val,
                                           Type type, Location loc) {
  return builder.create<ConstantOp>(loc, type, cast<TypedAttr>(val));
}

static RevertStrings getRevertStrings(ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<RevertStringsAttr>("sol.revert_strings"))
    return attr.getValue();
  return RevertStrings::Default;
}

bool mlir::sol::shouldEmitDebugRevertStrings(ModuleOp mod) {
  return getRevertStrings(mod) >= RevertStrings::Debug;
}

bool mlir::sol::shouldKeepUserRevertStrings(ModuleOp mod) {
  return getRevertStrings(mod) != RevertStrings::Strip;
}

bool mlir::sol::evmhasStaticCall(ModuleOp mod) {
  auto evmVersionAttr = cast<EvmVersionAttr>(mod->getAttr("sol.evm_version"));
  return evmVersionAttr.getValue() >= EvmVersion::Byzantium;
}

bool mlir::sol::evmSupportsReturnData(ModuleOp mod) {
  auto evmVersionAttr = cast<EvmVersionAttr>(mod->getAttr("sol.evm_version"));
  return evmVersionAttr.getValue() >= EvmVersion::Byzantium;
}

bool mlir::sol::evmCanOverchargeGasForCall(ModuleOp mod) {
  auto evmVersionAttr = cast<EvmVersionAttr>(mod->getAttr("sol.evm_version"));
  return evmVersionAttr.getValue() >= EvmVersion::TangerineWhistle;
}

Type mlir::sol::getEltType(Type ty, Index structTyIdx) {
  if (auto ptrTy = dyn_cast<sol::PointerType>(ty)) {
    return ptrTy.getPointeeType();
  }
  if (auto arrTy = dyn_cast<sol::ArrayType>(ty)) {
    return arrTy.getEltType();
  }
  if (isa<sol::StringType>(ty)) {
    return sol::FixedBytesType::get(ty.getContext(), /*size=*/1);
  }
  if (auto structTy = dyn_cast<sol::StructType>(ty)) {
    return structTy.getMemberTypes()[structTyIdx];
  }
  llvm_unreachable("Invalid type");
}

DataLocation mlir::sol::getDataLocation(Type ty) {
  return TypeSwitch<Type, DataLocation>(ty)
      .Case<PointerType>(
          [](sol::PointerType ptrTy) { return ptrTy.getDataLocation(); })
      .Case<ArrayType>(
          [](sol::ArrayType arrTy) { return arrTy.getDataLocation(); })
      .Case<StringType>(
          [](sol::StringType strTy) { return strTy.getDataLocation(); })
      .Case<StructType>(
          [](sol::StructType structTy) { return structTy.getDataLocation(); })
      .Case<MappingType>(
          [](sol::MappingType) { return sol::DataLocation::Storage; })
      .Default([&](Type) { return DataLocation::Stack; });
}

mlir::SideEffects::Resource *mlir::sol::getResource(DataLocation dataLoc) {
  switch (dataLoc) {
  case DataLocation::Stack:
    return StackResource::get();
  case DataLocation::CallData:
    return CallDataResource::get();
  case DataLocation::Memory:
    return MemoryResource::get();
  case DataLocation::Storage:
    return StorageResource::get();
  case DataLocation::Transient:
    return TransientResource::get();
  case DataLocation::Immutable:
    return ImmutableResource::get();
  }
}

// TODO? Should we exclude sol.pointer from reference types?

bool mlir::sol::isRefType(Type ty) {
  return isa<ArrayType>(ty) || isa<StringType>(ty) || isa<StructType>(ty) ||
         isa<PointerType>(ty) || isa<MappingType>(ty);
}

bool mlir::sol::isNonPtrRefType(Type ty) {
  return isRefType(ty) && !isa<PointerType>(ty);
}

bool mlir::sol::isLeftAligned(Type ty) {
  if (isa<IntegerType>(ty))
    return false;
  llvm_unreachable("NYI: isLeftAligned of other types");
}

bool mlir::sol::isDynamicallySized(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized();

  return false;
}

bool mlir::sol::hasDynamicallySizedElt(Type ty) {
  if (isa<StringType>(ty))
    return true;

  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.isDynSized() || hasDynamicallySizedElt(arrTy.getEltType());

  if (auto structTy = dyn_cast<StructType>(ty))
    return llvm::any_of(structTy.getMemberTypes(),
                        [](Type ty) { return hasDynamicallySizedElt(ty); });

  return false;
}

bool mlir::sol::isAddressLikeType(Type ty) {
  return isa<AddressType, ContractType>(ty);
}

unsigned mlir::sol::getStorageSlotCount(Type ty) {
  if (isa<IntegerType>(ty) || isa<EnumType>(ty) || isa<FixedBytesType>(ty) ||
      isa<MappingType>(ty) || isa<FuncRefType>(ty) || isa<ExtFuncRefType>(ty) ||
      isa<StringType>(ty) || isAddressLikeType(ty))
    return 1;

  if (auto arrTy = dyn_cast<ArrayType>(ty)) {
    // Dynamic arrays store only the head slot in-place.
    if (arrTy.isDynSized())
      return 1;

    Type eltTy = arrTy.getEltType();
    unsigned size = arrTy.getSize();
    if (!canBePacked(eltTy))
      return size * getStorageSlotCount(eltTy);

    // Packed arrays of small elements can fit in fewer slots.
    return llvm::divideCeil(size, 32u / getStorageByteSize(eltTy));
  }

  if (auto structTy = dyn_cast<StructType>(ty)) {
    assert(structTy.getDataLocation() == DataLocation::Storage &&
           "Storage slot count is only defined for storage structs");
    return structTy.getStorageSlotCount();
  }

  llvm_unreachable("NYI: Other types");
}

bool mlir::sol::canBePacked(Type ty) {
  // Scalars can be packed within a slot.
  if (isa<IntegerType>(ty) || isa<EnumType>(ty) || isa<FixedBytesType>(ty) ||
      isa<FuncRefType>(ty) || isa<ExtFuncRefType>(ty) || isAddressLikeType(ty))
    return true;

  // Aggregates are slot-aligned and cannot be packed.
  if (isa<ArrayType>(ty) || isa<StructType>(ty) || isa<MappingType>(ty) ||
      isa<StringType>(ty))
    return false;

  llvm_unreachable("NYI");
}

unsigned mlir::sol::getStorageByteSize(Type ty) {
  assert(canBePacked(ty) && "Only packable types have byte size");

  if (auto intTy = dyn_cast<IntegerType>(ty))
    // Bool occupies 1 byte in storage.
    return intTy.getWidth() == 1 ? 1 : intTy.getWidth() / 8;

  if (auto bytesTy = dyn_cast<FixedBytesType>(ty))
    return bytesTy.getSize();

  // Enums can have at most 256 members, so always 1 byte.
  if (isa<EnumType>(ty))
    return 1;

  // Address-like types are 20 bytes.
  if (isAddressLikeType(ty))
    return 20;

  // Internal function reference.
  if (isa<FuncRefType>(ty))
    return 8;

  // External function reference (address + selector).
  if (isa<ExtFuncRefType>(ty))
    return 20 + 4;

  llvm_unreachable("NYI");
}

static ParseResult parseDataLocation(AsmParser &parser,
                                     DataLocation &dataLocation) {
  StringRef dataLocationTok;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&dataLocationTok))
    return failure();

  auto parsedDataLoc = symbolizeDataLocation(dataLocationTok);
  if (!parsedDataLoc) {
    parser.emitError(loc, "Invalid data-location");
    return failure();
  }

  dataLocation = *parsedDataLoc;
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Parses a sol.array type.
///
///   array-type ::= `<` size `x` elt-ty `,` data-location `>`
///   size ::= fixed-size | `?`
///
Type ArrayType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  int64_t size = -1;
  if (parser.parseOptionalQuestion()) {
    if (parser.parseInteger(size))
      return {};
  }

  if (parser.parseKeyword("x"))
    return {};

  Type eleTy;
  if (parser.parseType(eleTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), size, eleTy, dataLocation);
}

/// Prints a sol.array type.
void ArrayType::print(AsmPrinter &printer) const {
  printer << "<";

  if (getSize() == -1)
    printer << "?";
  else
    printer << getSize();

  printer << " x " << getEltType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

/// Parses a sol.string type.
///
///   string-type ::= `<` data-location `>`
///
Type StringType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), dataLocation);
}

/// Prints a sol.string type.
void StringType::print(AsmPrinter &printer) const {
  printer << "<" << stringifyDataLocation(this->getDataLocation()) << ">";
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Parses a sol.struct type.
///
///   struct-type ::= `<` `(` member-types `)` `,` data-location `>`
///
Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  if (parser.parseLParen())
    return {};

  SmallVector<Type, 4> memTys;
  do {
    Type memTy;
    if (parser.parseType(memTy))
      return {};
    memTys.push_back(memTy);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), memTys, dataLocation);
}

/// Prints a sol.array type.
void StructType::print(AsmPrinter &printer) const {
  printer << "<(";
  llvm::interleaveComma(getMemberTypes(), printer.getStream(),
                        [&](Type memTy) { printer << memTy; });
  printer << "), " << stringifyDataLocation(getDataLocation()) << ">";
}

static void computeStructStorageMemberOffsets(
    ArrayRef<Type> memberTypes, SmallVectorImpl<uint64_t> &slotOffsets,
    SmallVectorImpl<uint64_t> &byteOffsets, uint64_t &storageSlotCount) {
  uint64_t slotOffset = 0;
  uint64_t byteOffset = 0;

  slotOffsets.reserve(memberTypes.size());
  byteOffsets.reserve(memberTypes.size());

  for (Type memberTy : memberTypes) {
    if (canBePacked(memberTy)) {
      uint64_t memberByteSize = getStorageByteSize(memberTy);
      if (byteOffset + memberByteSize > 32) {
        ++slotOffset;
        byteOffset = 0;
      }

      slotOffsets.push_back(slotOffset);
      byteOffsets.push_back(byteOffset);
      byteOffset += memberByteSize;
      continue;
    }

    if (byteOffset != 0) {
      ++slotOffset;
      byteOffset = 0;
    }
    slotOffsets.push_back(slotOffset);
    byteOffsets.push_back(0);
    slotOffset += getStorageSlotCount(memberTy);
  }

  if (byteOffset > 0)
    ++slotOffset;

  storageSlotCount = slotOffset;
}

StructType::StorageMemberOffset
StructType::getStorageMemberOffset(uint64_t memberIdx) const {
  assert(getDataLocation() == DataLocation::Storage &&
         "Storage offsets are only defined for storage structs");
  assert(memberIdx < getMemberTypes().size() && "Member index out of bounds");

  return {/*slotOffset=*/getMemberSlotOffsets()[memberIdx],
          /*byteOffset=*/getMemberByteOffsets()[memberIdx]};
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

/// Parses a sol.ptr type.
///
///   ptr-type ::= `<` pointee-ty, data-location `>`
///
Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type pointeeTy;
  if (parser.parseType(pointeeTy))
    return {};

  if (parser.parseComma())
    return {};

  DataLocation dataLocation = DataLocation::Memory;
  if (parseDataLocation(parser, dataLocation))
    return {};

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), pointeeTy, dataLocation);
}

/// Prints a sol.ptr type.
void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType() << ", "
          << stringifyDataLocation(getDataLocation()) << ">";
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Sol/SolOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Sol/SolOpsTypes.cpp.inc"

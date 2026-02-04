//===- EVMConstants.h - EVM constants and utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EVM-specific constants and utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SOLTOSTANDARD_EVMCONSTANTS_H
#define MLIR_CONVERSION_SOLTOSTANDARD_EVMCONSTANTS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/KECCAK.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace evm {

/// Panic codes as defined in Solidity.
enum class PanicCode : uint8_t {
  Generic = 0x00,              // generic / unspecified error
  Assert = 0x01,               // used by the assert() builtin
  UnderOverflow = 0x11,        // arithmetic underflow or overflow
  DivisionByZero = 0x12,       // division or modulo by zero
  EnumConversionError = 0x21,  // enum conversion error
  StorageEncodingError = 0x22, // invalid encoding in storage
  EmptyArrayPop = 0x31,        // empty array pop
  ArrayOutOfBounds = 0x32,     // array out of bounds access
  ResourceError =
      0x41, // resource error (too large allocation or too large array)
  InvalidInternalFunction = 0x51, // calling invalid internal function
};

/// Solidity memory layout constants.
namespace MemoryLayout {
/// The free pointer location.
constexpr size_t freeMemoryPointer = 64;
/// The zero pointer location.
constexpr size_t zeroPointer = freeMemoryPointer + 32;
/// Start of general purpose memory.
constexpr size_t generalPurposeMemoryStart = zeroPointer + 32;
} // namespace MemoryLayout

/// Computes the keccak256 hash of the input string.
inline std::array<uint8_t, 32> keccak256(llvm::StringRef input) {
  return llvm::KECCAK::KECCAK_256(input);
}

/// Computes the ABI selector (first 4 bytes of keccak256) for a function
/// signature.
inline uint32_t selectorFromSignature(llvm::StringRef signature) {
  auto hash = keccak256(signature);
  return (static_cast<uint32_t>(hash[0]) << 24) |
         (static_cast<uint32_t>(hash[1]) << 16) |
         (static_cast<uint32_t>(hash[2]) << 8) | static_cast<uint32_t>(hash[3]);
}

/// Computes the ABI selector as a 256-bit value (left-aligned).
inline llvm::APInt selectorFromSignatureU256(llvm::StringRef signature) {
  uint32_t selector = selectorFromSignature(signature);
  llvm::APInt result(256, selector);
  return result << (256 - 32);
}

/// Computes keccak256 and returns the result as a 256-bit APInt.
inline llvm::APInt keccak256AsAPInt(llvm::StringRef input) {
  auto hash = keccak256(input);
  // Convert 32 bytes to APInt (big-endian)
  llvm::APInt result(256, 0);
  for (int i = 0; i < 32; ++i) {
    result <<= 8;
    result |= hash[i];
  }
  return result;
}

} // namespace evm
} // namespace mlir

#endif // MLIR_CONVERSION_SOLTOSTANDARD_EVMCONSTANTS_H

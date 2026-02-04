//===- Target.h - MLIR Solidity Target definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target enum for Solidity compilation targets.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SOL_TARGET_H_
#define MLIR_DIALECT_SOL_TARGET_H_

#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace sol {

enum class Target {
  EVM,
  Undefined,
};

/// Return the enum Target from the string (case insensitive).
Target strToTarget(llvm::StringRef str);

} // namespace sol
} // namespace mlir

#endif // MLIR_DIALECT_SOL_TARGET_H_

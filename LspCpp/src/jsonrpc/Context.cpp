//===--- Context.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibLsp/JsonRpc/Context.h"
#include <cassert>

namespace lsp {


Context Context::empty() { return Context(/*dataPtr=*/nullptr); }

Context::Context(std::shared_ptr<const Data> DataPtr)
    : dataPtr(std::move(DataPtr)) {}

Context Context::clone() const { return Context(dataPtr); }

static Context &currentContext() {
  static thread_local auto c = Context::empty();
  return c;
}

const Context &Context::current() { return currentContext(); }

Context Context::swapCurrent(Context Replacement) {
  std::swap(Replacement, currentContext());
  return Replacement;
}


} // lsp clang

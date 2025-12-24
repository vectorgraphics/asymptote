#pragma once

#include "asyffiimpl.h"
#include "common.h"
#include "vm.h"

namespace camp
{
IAsyContext* getAsyContext();
}

record* loadDynLib(string const& key, string const& fileName);

string tryGetDllPath(string const& libName);

void unloadLib(string const& key);

void callForeignFunction(vm::stack* stack, TAsyForeignFunction fn);

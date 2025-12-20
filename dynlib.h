#pragma once

#include "asyffiimpl.h"
#include "common.h"
#include "vm.h"

namespace camp
{
IAsyContext* getAsyContext();
}


void loadDynLib(string const& key);

void unloadLib(string const& key);

void callFunction1(vm::stack* stack);
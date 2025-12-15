#pragma once

#include "asyffiimpl.h"
#include "common.h"


void loadDynLib(string const& key);

void callFunction(
        string const& key, string const& fnName, camp::AsyArgsImpl* args
);

void unloadLib(string const& key);
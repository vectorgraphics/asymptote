#pragma once

#include "common.h"


void loadDynLib(string const& key);

void callFunction(string const& key, string const& fnName);

void unloadLib(string const& key);
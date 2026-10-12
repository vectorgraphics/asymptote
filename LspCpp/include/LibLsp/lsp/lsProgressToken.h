#pragma once

#include "LibLsp/JsonRpc/serializer.h"

#include <string>
#include <utility>

/**
 * The LSP progress token used by WorkDoneProgressParams/PartialResultParams:
 * ProgressToken = integer | string.
 */
using lsProgressToken = std::pair<optional<std::string>, optional<int>>;

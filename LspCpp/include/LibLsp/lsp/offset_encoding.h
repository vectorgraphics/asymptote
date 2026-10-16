#pragma once

#include "LibLsp/lsp/lsPosition.h"
#include "LibLsp/lsp/utils.h"

#include <string>

namespace lsp
{

enum class OffsetEncoding
{
    Utf16,
    Utf8,
};

struct OffsetEncodingContext
{
    OffsetEncoding encoding = OffsetEncoding::Utf16;
};

// Opt-in position helpers. WorkingFiles and utils::GetOffsetForPosition keep UTF-16 defaults.
int GetOffsetForPositionWithEncoding(lsPosition position, std::string const& content, OffsetEncoding encoding);
lsPosition GetPositionForOffsetWithEncoding(size_t offset, std::string const& content, OffsetEncoding encoding);

} // namespace lsp

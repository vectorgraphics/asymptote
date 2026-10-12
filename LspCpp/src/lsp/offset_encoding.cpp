#include "LibLsp/lsp/offset_encoding.h"

namespace lsp
{

int GetOffsetForPositionWithEncoding(lsPosition position, std::string const& content, OffsetEncoding encoding)
{
    if (encoding == OffsetEncoding::Utf16)
    {
        return GetOffsetForPosition(position, content);
    }

    size_t i = 0;
    while (position.line > 0 && i < content.size())
    {
        if (content[i] == '\n')
        {
            --position.line;
        }
        ++i;
    }
    return static_cast<int>(i + static_cast<size_t>(position.character));
}

lsPosition GetPositionForOffsetWithEncoding(size_t offset, std::string const& content, OffsetEncoding encoding)
{
    if (encoding == OffsetEncoding::Utf16)
    {
        return GetPositionForOffset(offset, content);
    }

    lsPosition result;
    size_t line_start = 0;
    for (size_t i = 0; i < offset && i < content.size(); ++i)
    {
        if (content[i] == '\n')
        {
            ++result.line;
            line_start = i + 1;
        }
    }
    result.character = static_cast<int>(offset - line_start);
    return result;
}

} // namespace lsp

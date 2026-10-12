#pragma once

#include "LibLsp/JsonRpc/Context.h"
#include "LibLsp/lsp/offset_encoding.h"
#include "LibLsp/lsp/path_mapping.h"

#include <memory>

namespace lsp
{

extern Key<OffsetEncodingContext> offsetEncodingContextKey;
extern Key<std::shared_ptr<PathMapping const>> pathMappingContextKey;

OffsetEncoding currentOffsetEncoding(Context const& ctx = Context::current());
OffsetEncodingContext currentOffsetEncodingContext(Context const& ctx = Context::current());

/// Returns a child context carrying the requested offset encoding.
Context withOffsetEncodingContext(OffsetEncoding encoding);
Context withOffsetEncodingContext(OffsetEncodingContext context);

std::shared_ptr<PathMapping const> currentPathMapping(Context const& ctx = Context::current());
Context withPathMapping(std::shared_ptr<PathMapping const> mapping);

int GetOffsetForPositionInContext(
    lsPosition position,
    std::string const& content,
    Context const& ctx = Context::current()
);
lsPosition GetPositionForOffsetInContext(
    size_t offset,
    std::string const& content,
    Context const& ctx = Context::current()
);

} // namespace lsp

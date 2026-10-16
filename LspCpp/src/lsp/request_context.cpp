#include "LibLsp/lsp/request_context.h"

namespace lsp
{

Key<OffsetEncodingContext> offsetEncodingContextKey;
Key<std::shared_ptr<PathMapping const>> pathMappingContextKey;

OffsetEncoding currentOffsetEncoding(Context const& ctx)
{
    return currentOffsetEncodingContext(ctx).encoding;
}

OffsetEncodingContext currentOffsetEncodingContext(Context const& ctx)
{
    if (auto const* value = ctx.get(offsetEncodingContextKey))
    {
        return *value;
    }
    return OffsetEncodingContext {};
}

Context withOffsetEncodingContext(OffsetEncoding encoding)
{
    OffsetEncodingContext context;
    context.encoding = encoding;
    return withOffsetEncodingContext(context);
}

Context withOffsetEncodingContext(OffsetEncodingContext context)
{
    return Context::current().derive(offsetEncodingContextKey, std::move(context));
}

std::shared_ptr<PathMapping const> currentPathMapping(Context const& ctx)
{
    if (auto const* value = ctx.get(pathMappingContextKey))
    {
        return *value;
    }
    return {};
}

Context withPathMapping(std::shared_ptr<PathMapping const> mapping)
{
    return Context::current().derive(pathMappingContextKey, std::move(mapping));
}

int GetOffsetForPositionInContext(lsPosition position, std::string const& content, Context const& ctx)
{
    return GetOffsetForPositionWithEncoding(position, content, currentOffsetEncoding(ctx));
}

lsPosition GetPositionForOffsetInContext(size_t offset, std::string const& content, Context const& ctx)
{
    return GetPositionForOffsetWithEncoding(offset, content, currentOffsetEncoding(ctx));
}

} // namespace lsp

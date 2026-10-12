#include "LibLsp/lsp/path_mapping.h"

#include "LibLsp/lsp/uri.h"
#include "LibLsp/lsp/utils.h"

#include <utility>

namespace lsp
{
namespace
{
bool IsPathSeparator(char c)
{
    return c == '/' || c == '\\';
}

bool MatchesPathPrefix(std::string const& value, std::string const& prefix)
{
    if (!StartsWith(value, prefix))
    {
        return false;
    }
    if (value.size() == prefix.size())
    {
        return true;
    }
    if (prefix.empty())
    {
        return false;
    }
    if (IsPathSeparator(prefix.back()))
    {
        return true;
    }
    return IsPathSeparator(value[prefix.size()]);
}

std::string RemapPrefix(std::string const& value, std::string const& from_prefix, std::string const& to_prefix)
{
    if (!MatchesPathPrefix(value, from_prefix))
    {
        return value;
    }
    return to_prefix + value.substr(from_prefix.size());
}
} // namespace

void PathMapping::add(std::string client_prefix, std::string server_prefix)
{
    entries_.push_back(PathMappingEntry {std::move(client_prefix), std::move(server_prefix)});
}

std::string PathMapping::client_to_server(std::string const& client_path) const
{
    for (auto const& entry : entries_)
    {
        auto remapped = RemapPrefix(client_path, entry.client_prefix, entry.server_prefix);
        if (remapped != client_path)
        {
            return remapped;
        }
    }
    return client_path;
}

std::string PathMapping::server_to_client(std::string const& server_path) const
{
    for (auto const& entry : entries_)
    {
        auto remapped = RemapPrefix(server_path, entry.server_prefix, entry.client_prefix);
        if (remapped != server_path)
        {
            return remapped;
        }
    }
    return server_path;
}

lsDocumentUri PathMapping::remap_uri_to_server(lsDocumentUri const& client_uri) const
{
    Uri uri = Uri::from_raw(client_uri.raw_uri_);
    if (uri.is_test_scheme())
    {
        return lsDocumentUri(uri.to_absolute_path());
    }
    return lsDocumentUri(AbsolutePath::FromNormalized(client_to_server(client_uri.GetRawPath())));
}

lsDocumentUri PathMapping::remap_uri_to_client(std::string const& server_path) const
{
    return lsDocumentUri(AbsolutePath::FromNormalized(server_to_client(server_path)));
}

} // namespace lsp

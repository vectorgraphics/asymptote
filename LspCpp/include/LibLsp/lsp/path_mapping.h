#pragma once

#include "LibLsp/lsp/lsDocumentUri.h"

#include <string>
#include <vector>

namespace lsp
{

struct PathMappingEntry
{
    std::string client_prefix;
    std::string server_prefix;
};

// Opt-in client/server path remapping for remote or devcontainer setups.
class PathMapping
{
public:
    void add(std::string client_prefix, std::string server_prefix);

    std::string client_to_server(std::string const& client_path) const;
    std::string server_to_client(std::string const& server_path) const;

    lsDocumentUri remap_uri_to_server(lsDocumentUri const& client_uri) const;
    lsDocumentUri remap_uri_to_client(std::string const& server_path) const;

private:
    std::vector<PathMappingEntry> entries_;
};

} // namespace lsp

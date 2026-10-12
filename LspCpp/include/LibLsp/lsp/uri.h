#pragma once

#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/lsDocumentUri.h"

#include <string>

namespace lsp
{

// Opt-in URI helpers. Existing lsDocumentUri / make_file_scheme_uri behavior is unchanged.
struct Uri
{
    std::string raw;

    static Uri from_raw(std::string uri);
    static Uri from_file_path(AbsolutePath const& path);
    static Uri from_test_path(std::string path);

    std::string scheme() const;
    std::string body() const;
    AbsolutePath to_absolute_path() const;
    bool is_test_scheme() const;
};

std::string percent_encode_uri_component(std::string const& value);
std::string percent_decode_uri_component(std::string const& value);

} // namespace lsp

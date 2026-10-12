#include "LibLsp/lsp/uri.h"

#include "LibLsp/lsp/utils.h"

#include <cctype>
#include <utility>

namespace lsp
{
namespace
{
bool IsUnreserved(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '~';
}

bool IsTestPathSafe(char c)
{
    return IsUnreserved(c) || c == '/' || c == ':';
}

std::string NormalizeTestPath(std::string path)
{
    for (char& c : path)
    {
        if (c == '\\')
        {
            c = '/';
        }
    }
    while (!path.empty() && path[0] == '/')
    {
        path.erase(path.begin());
    }
    return path;
}

std::string PercentEncodeTestPath(std::string const& value)
{
    static char const* hex = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(value.size());
    for (unsigned char c : value)
    {
        if (IsTestPathSafe(static_cast<char>(c)))
        {
            encoded.push_back(static_cast<char>(c));
        }
        else
        {
            encoded.push_back('%');
            encoded.push_back(hex[c >> 4]);
            encoded.push_back(hex[c & 0xF]);
        }
    }
    return encoded;
}
} // namespace

Uri Uri::from_raw(std::string uri)
{
    Uri result;
    result.raw = std::move(uri);
    return result;
}

Uri Uri::from_file_path(AbsolutePath const& path)
{
    Uri result;
    result.raw = make_file_scheme_uri(path.path());
    return result;
}

Uri Uri::from_test_path(std::string path)
{
    Uri result;
    path = NormalizeTestPath(std::move(path));
    result.raw = "test:///" + PercentEncodeTestPath(path);
    return result;
}

std::string Uri::scheme() const
{
    auto const pos = raw.find(':');
    if (pos == std::string::npos)
    {
        return {};
    }
    return raw.substr(0, pos);
}

std::string Uri::body() const
{
    auto const pos = raw.find("://");
    if (pos == std::string::npos)
    {
        return raw;
    }
    return raw.substr(pos + 3);
}

AbsolutePath Uri::to_absolute_path() const
{
    if (scheme() == "file")
    {
        lsDocumentUri doc_uri;
        doc_uri.raw_uri_ = raw;
        return doc_uri.GetAbsolutePath();
    }
    if (scheme() == "test")
    {
        std::string path = percent_decode_uri_component(body());
        if (path.empty() || path[0] != '/')
        {
            path.insert(path.begin(), '/');
        }
        return AbsolutePath(path);
    }
    return AbsolutePath();
}

bool Uri::is_test_scheme() const
{
    return scheme() == "test";
}

std::string percent_encode_uri_component(std::string const& value)
{
    static char const* hex = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(value.size());
    for (unsigned char c : value)
    {
        if (IsUnreserved(static_cast<char>(c)))
        {
            encoded.push_back(static_cast<char>(c));
        }
        else
        {
            encoded.push_back('%');
            encoded.push_back(hex[c >> 4]);
            encoded.push_back(hex[c & 0xF]);
        }
    }
    return encoded;
}

std::string percent_decode_uri_component(std::string const& value)
{
    std::string decoded;
    decoded.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '%' && i + 2 < value.size())
        {
            auto hex = [](char c) -> int
            {
                if (c >= '0' && c <= '9')
                {
                    return c - '0';
                }
                if (c >= 'A' && c <= 'F')
                {
                    return c - 'A' + 10;
                }
                if (c >= 'a' && c <= 'f')
                {
                    return c - 'a' + 10;
                }
                return -1;
            };
            int const hi = hex(value[i + 1]);
            int const lo = hex(value[i + 2]);
            if (hi >= 0 && lo >= 0)
            {
                decoded.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        decoded.push_back(value[i]);
    }
    return decoded;
}

} // namespace lsp

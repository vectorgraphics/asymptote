#include "LibLsp/lsp/offset_encoding.h"
#include "LibLsp/lsp/path_mapping.h"
#include "LibLsp/lsp/uri.h"
#include "test_helpers.h"

#include <string>

namespace
{
using test::Expect;

void TestTestUriRoundTrip()
{
    lsp::Uri uri = lsp::Uri::from_test_path("foo dir/bar#baz?.cpp");
    Expect(uri.scheme() == "test", "test URI must use test scheme");
    Expect(
        uri.raw == "test:///foo%20dir/bar%23baz%3F.cpp",
        "test URI must percent-encode non-path characters");
    Expect(uri.to_absolute_path().path() == "/foo dir/bar#baz?.cpp", "test URI must map to absolute server path");
}

void TestWindowsTestUriRoundTrip()
{
    lsp::Uri uri = lsp::Uri::from_test_path(R"(C:\client dir\main.cpp)");
    Expect(uri.raw == "test:///C:/client%20dir/main.cpp", "test URI must normalize Windows separators");
    Expect(uri.to_absolute_path().path() == "/C:/client dir/main.cpp", "Windows test URI must round-trip path text");
}

void TestFileUriUsesExistingBehavior()
{
    AbsolutePath path = AbsolutePath::FromNormalized("/workspace/main.cpp");
    lsp::Uri uri = lsp::Uri::from_file_path(path);
    Expect(uri.scheme() == "file", "file URI must keep file scheme");
    Expect(uri.to_absolute_path() == path, "file URI round-trip must preserve absolute path");
}

void TestPathMappingRemapsClientAndServerPaths()
{
    lsp::PathMapping mapping;
    mapping.add("/client/project", "/server/project");

    Expect(
        mapping.client_to_server("/client/project/src/a.cpp") == "/server/project/src/a.cpp",
        "client_to_server must remap matching prefix");
    Expect(
        mapping.server_to_client("/server/project/src/a.cpp") == "/client/project/src/a.cpp",
        "server_to_client must remap matching prefix");
    Expect(
        mapping.client_to_server("/client/projectile/src/a.cpp") == "/client/projectile/src/a.cpp",
        "client_to_server must not remap adjacent directory names");
    Expect(
        mapping.server_to_client("/server/projectile/src/a.cpp") == "/server/projectile/src/a.cpp",
        "server_to_client must not remap adjacent directory names");
}

void TestPathMappingRemapsDocumentUris()
{
    lsp::PathMapping mapping;
    mapping.add("/client/project", "/server/project");

    lsDocumentUri client_uri;
    client_uri.raw_uri_ = "file:///client/project/src/main.cpp";
    auto const server_uri = mapping.remap_uri_to_server(client_uri);
    Expect(
        server_uri.GetRawPath() == "/server/project/src/main.cpp",
        "remap_uri_to_server must translate client URI path");
}

void TestOffsetEncodingUtf8Columns()
{
    std::string const content = "/*\xc3\xb6*/int x;\nint y=x;";
    lsPosition utf16_pos;
    utf16_pos.line = 0;
    utf16_pos.character = 9;
    lsPosition utf8_pos;
    utf8_pos.line = 0;
    utf8_pos.character = 11;

    Expect(
        lsp::GetOffsetForPositionWithEncoding(utf16_pos, content, lsp::OffsetEncoding::Utf16) ==
            lsp::GetOffsetForPosition(utf16_pos, content),
        "UTF-16 encoding helper must match existing default offset semantics");
    Expect(
        lsp::GetOffsetForPositionWithEncoding(utf8_pos, content, lsp::OffsetEncoding::Utf8) == 11,
        "UTF-8 encoding helper must count UTF-8 code units on multibyte lines");
}

} // namespace

int main(int argc, char** argv)
{
    test::InitTestFilter(argc, argv);
    RUN_TEST(TestTestUriRoundTrip);
    RUN_TEST(TestWindowsTestUriRoundTrip);
    RUN_TEST(TestFileUriUsesExistingBehavior);
    RUN_TEST(TestPathMappingRemapsClientAndServerPaths);
    RUN_TEST(TestPathMappingRemapsDocumentUris);
    RUN_TEST(TestOffsetEncodingUtf8Columns);
    return test::Failures() == 0 ? 0 : 1;
}

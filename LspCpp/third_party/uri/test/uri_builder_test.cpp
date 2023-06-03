// Copyright (c) Glyn Matthews 2012-2016.
// Copyright 2012 Google, Inc.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <gtest/gtest.h>
#include <network/uri.hpp>
#include "string_utility.hpp"

TEST(builder_test, empty_uri_doesnt_throw) {
  network::uri_builder builder;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, empty_uri) {
  network::uri_builder builder;
  network::uri instance(builder);
  ASSERT_TRUE(instance.empty());
}

TEST(builder_test, simple_uri_doesnt_throw) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, simple_uri) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com/", builder.uri().string());
}

TEST(builder_test, simple_uri_has_scheme) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_TRUE(builder.uri().has_scheme());
}

TEST(builder_test, simple_uri_scheme_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("http", builder.uri().scheme());
}

TEST(builder_test, simple_uri_has_no_user_info) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_FALSE(builder.uri().has_user_info());
}

TEST(builder_test, simple_uri_has_host) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_TRUE(builder.uri().has_host());
}

TEST(builder_test, simple_uri_host_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("www.example.com", builder.uri().host());
}

TEST(builder_test, simple_uri_has_no_port) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_FALSE(builder.uri().has_port());
}

TEST(builder_test, simple_uri_has_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_TRUE(builder.uri().has_path());
}

TEST(builder_test, simple_uri_path_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("/", builder.uri().path());
}

TEST(builder_test, simple_uri_has_no_query) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_FALSE(builder.uri().has_query());
}

TEST(builder_test, simple_uri_has_no_fragment) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_FALSE(builder.uri().has_fragment());
}

TEST(builder_test, simple_opaque_uri_doesnt_throw) {
  network::uri_builder builder;
  builder
    .scheme("mailto")
    .path("john.doe@example.com")
    ;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, simple_opaque_uri) {
  network::uri_builder builder;
  builder
    .scheme("mailto")
    .path("john.doe@example.com")
    ;
  ASSERT_EQ("mailto:john.doe@example.com", builder.uri().string());
}

TEST(builder_test, simple_opaque_uri_has_scheme) {
  network::uri_builder builder;
  builder
    .scheme("mailto")
    .path("john.doe@example.com")
    ;
  ASSERT_TRUE(builder.uri().has_scheme());
}

TEST(builder_test, simple_opaque_uri_scheme_value) {
  network::uri_builder builder;
  builder
    .scheme("mailto")
    .path("john.doe@example.com")
    ;
  ASSERT_EQ("mailto", builder.uri().scheme());
}

TEST(builder_test, relative_hierarchical_uri_doesnt_throw) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, relative_hierarchical_uri) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("www.example.com/", builder.uri().string());
}

TEST(builder_test, relative_opaque_uri_doesnt_throw) {
  network::uri_builder builder;
  builder
    .path("john.doe@example.com")
    ;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, relative_opaque_uri) {
  network::uri_builder builder;
  builder
    .path("john.doe@example.com")
    ;
  ASSERT_EQ("john.doe@example.com", builder.uri().string());
}

TEST(builder_test, full_uri_doesnt_throw) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_NO_THROW(builder.uri());
}

TEST(builder_test, full_uri) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("http://user@www.example.com:80/path?query=value#fragment", builder.uri().string());
}

TEST(builder_test, full_uri_has_scheme) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_scheme());
}

TEST(builder_test, full_uri_scheme_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("http", builder.uri().scheme());
}

TEST(builder_test, full_uri_has_user_info) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_user_info());
}

TEST(builder_test, full_uri_user_info_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("user", builder.uri().user_info());
}

TEST(builder_test, full_uri_has_host) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_host());
}

TEST(builder_test, full_uri_host_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("www.example.com", builder.uri().host());
}

TEST(builder_test, full_uri_has_port) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_port());
}

TEST(builder_test, full_uri_has_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_path());
}

TEST(builder_test, full_uri_path_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("/path", builder.uri().path());
}

TEST(builder_test, full_uri_has_query) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_query());
}

TEST(builder_test, full_uri_query_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("query=value", builder.uri().query());
}

TEST(builder_test, full_uri_has_fragment) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_TRUE(builder.uri().has_fragment());
}

TEST(builder_test, full_uri_fragment_value) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .user_info("user")
    .host("www.example.com")
    .port("80")
    .path("/path")
    .append_query_key_value_pair("query", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("fragment", builder.uri().fragment());
}

TEST(builder_test, relative_uri) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("www.example.com/", builder.uri().string());
}

TEST(builder_test, relative_uri_scheme) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_FALSE(builder.uri().has_scheme());
}

TEST(builder_test, authority) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .authority("www.example.com:8080")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com:8080/", builder.uri().string());
  ASSERT_EQ("www.example.com", builder.uri().host());
  ASSERT_EQ("8080", builder.uri().port());
}

TEST(builder_test, relative_uri_has_host) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_TRUE(builder.uri().has_host());
}

TEST(builder_test, relative_uri_host_value) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("www.example.com", builder.uri().host());
}

TEST(builder_test, relative_uri_has_path) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_TRUE(builder.uri().has_path());
}

TEST(builder_test, relative_uri_path_value) {
  network::uri_builder builder;
  builder
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("/", builder.uri().path());
}

TEST(builder_test, build_relative_uri_with_path_query_and_fragment) {
  network::uri_builder builder;
  builder
    .path("/path/")
    .append_query_key_value_pair("key", "value")
    .fragment("fragment")
    ;
  ASSERT_EQ("/path/", builder.uri().path());
  ASSERT_EQ("key=value", builder.uri().query());
  ASSERT_EQ("fragment", builder.uri().fragment());
}

TEST(builder_test, build_uri_with_capital_scheme) {
  network::uri_builder builder;
  builder
    .scheme("HTTP")
    .host("www.example.com")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com/", builder.uri().string());
}

TEST(builder_test, build_uri_with_capital_host) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("WWW.EXAMPLE.COM")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com/", builder.uri().string());
}

TEST(builder_test, build_uri_with_unencoded_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/A path with spaces")
    ;
  ASSERT_EQ("http://www.example.com/A%20path%20with%20spaces", builder.uri().string());
}

TEST(builder_test, DISABLED_builder_uri_and_remove_dot_segments_from_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/A/./path/")
    ;
  ASSERT_EQ("http://www.example.com/A/path/", builder.uri().string());
}

TEST(builder_test, build_uri_with_qmark_in_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/?/")
    ;
  ASSERT_EQ("http://www.example.com/%3F/", builder.uri().string());
}

TEST(builder_test, build_uri_with_hash_in_path) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .path("/#/")
    ;
  ASSERT_EQ("http://www.example.com/%23/", builder.uri().string());
}

TEST(builder_test, simple_port) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .port(8000)
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com:8000/", builder.uri().string());
}

TEST(builder_test, build_uri_with_query_item) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .append_query_key_value_pair("a", "1")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com/?a=1", builder.uri().string());
}

TEST(builder_test, build_uri_with_multiple_query_items) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("www.example.com")
    .append_query_key_value_pair("a", "1")
    .append_query_key_value_pair("b", "2")
    .path("/")
    ;
  ASSERT_EQ("http://www.example.com/?a=1&b=2", builder.uri().string());
}

TEST(builder_test, build_uri_with_query_item_with_encoded_chars)
{
    network::uri_builder builder;
    builder
    .scheme("http")
    .host("www.example.com")
    .append_query_key_value_pair("a", "parameter with encoded chars!")
    .path("/")
    ;
    ASSERT_EQ("http://www.example.com/?a=parameter%20with%20encoded%20chars%21", builder.uri().string());
}

TEST(builder_test, build_uri_with_multiple_query_items_with_encoded_chars) {
    network::uri_builder builder;
    builder
    .scheme("http")
    .host("www.example.com")
    .append_query_key_value_pair("a", "first parameter with encoded chars!")
    .append_query_key_value_pair("b", "second parameter with encoded chars!")
    .path("/")
    ;
    ASSERT_EQ("http://www.example.com/?a=first%20parameter%20with%20encoded%20chars%21&b=second%20parameter%20with%20encoded%20chars%21", builder.uri().string());
}

TEST(builder_test, construct_from_existing_uri) {
  network::uri instance("http://www.example.com/");
  network::uri_builder builder(instance);
  ASSERT_EQ("http://www.example.com/", builder.uri().string());
}

TEST(builder_test, build_from_existing_uri) {
  network::uri instance("http://www.example.com/");
  network::uri_builder builder(instance);
  builder.append_query_key_value_pair("a", "1").append_query_key_value_pair("b", "2").fragment("fragment");
  ASSERT_EQ("http://www.example.com/?a=1&b=2#fragment", builder.uri().string());
}

TEST(builder_test, authority_without_port_test) {
  network::uri_builder builder;
  builder
    .scheme("https")
    .authority("www.example.com")
    ;
  ASSERT_EQ("www.example.com", builder.uri().authority());
}

TEST(builder_test, authority_with_port_test) {
  network::uri_builder builder;
  builder
    .scheme("https")
    .authority("www.example.com:")
    ;
  ASSERT_EQ("www.example.com:", builder.uri().authority());
}

TEST(builder_test, DISABLED_authority_without_host_test) {
  network::uri_builder builder;
  builder
    .scheme("https")
    .authority(":1234")
    ;
  ASSERT_EQ(":1234", builder.uri().authority());
}

TEST(builder_test, clear_user_info_test) {
  network::uri instance("http://user@www.example.com:80/path?query#fragment");
  network::uri_builder builder(instance);
  builder.clear_user_info();
  ASSERT_EQ("http://www.example.com:80/path?query#fragment", builder.uri().string());
}

TEST(builder_test, clear_port_test) {
  network::uri instance("http://user@www.example.com:80/path?query#fragment");
  network::uri_builder builder(instance);
  builder.clear_port();
  ASSERT_EQ("http://user@www.example.com/path?query#fragment", builder.uri().string());
}

TEST(builder_test, clear_path_test) {
  network::uri instance("http://user@www.example.com:80/path?query#fragment");
  network::uri_builder builder(instance);
  builder.clear_path();
  ASSERT_EQ("http://user@www.example.com:80?query#fragment", builder.uri().string());
}

TEST(builder_test, clear_query_test) {
  network::uri instance("http://user@www.example.com:80/path?query#fragment");
  network::uri_builder builder(instance);
  builder.clear_query();
  ASSERT_EQ("http://user@www.example.com:80/path#fragment", builder.uri().string());
}

TEST(uri_test, clear_query_params_with_no_query) {
	network::uri original("http://example.com/path");
	network::uri_builder builder(original);
	builder.clear_query();
}

TEST(builder_test, clear_fragment_test) {
  network::uri instance("http://user@www.example.com:80/path?query#fragment");
  network::uri_builder builder(instance);
  builder.clear_fragment();
  ASSERT_EQ("http://user@www.example.com:80/path?query", builder.uri().string());
}

TEST(builder_test, empty_username) {
  std::string user_info(":");
  network::uri_builder builder;
  builder.scheme("ftp").host("127.0.0.1").user_info(user_info);
  ASSERT_EQ("ftp://:@127.0.0.1", builder.uri().string());
}

TEST(builder_test, path_should_be_prefixed_with_slash) {
  std::string path("relative");
  network::uri_builder builder;
  builder.scheme("ftp").host("127.0.0.1").path(path);
  ASSERT_EQ("ftp://127.0.0.1/relative", builder.uri().string());
}

TEST(builder_test, path_should_be_prefixed_with_slash_2) {
  network::uri_builder builder;
  builder
    .scheme("ftp").host("127.0.0.1").path("noleadingslash/foo.txt");
  ASSERT_EQ("/noleadingslash/foo.txt", builder.uri().path());
}

TEST(builder_test, set_multiple_query_with_encoding) {
  network::uri_builder builder;
  builder
    .scheme("http")
    .host("example.com")
    .append_query_key_value_pair("q1", "foo bar")
    .append_query_key_value_pair("q2", "biz baz")
    ;
  ASSERT_EQ("http://example.com?q1=foo%20bar&q2=biz%20baz", builder.uri().string());
}

TEST(builder_test, non_array_string_literals_should_work) {
  const char* p = "http";
  const char* q = "foo";

  network::uri_builder builder;
  builder
    .scheme(p)
    .host("example.com")
    .path(q)
    ;
  ASSERT_EQ("http://example.com/foo", builder.uri());
}

TEST(builder_test, non_const_non_array_string_literals_should_work) {
  const char* p = "http";
  const char* q = "foo";

  network::uri_builder builder;
  builder
    .scheme(const_cast<char *>(p))
    .host("example.com")
    .path(const_cast<char *>(q))
    ;
  ASSERT_EQ("http://example.com/foo", builder.uri());
}

TEST(builder_test, scheme_and_absolute_path) {
  network::uri_builder builder;
  builder
    .scheme("foo")
    .path("/bar")
    ;
  ASSERT_EQ("foo:/bar", builder.uri());
  ASSERT_EQ("foo", builder.uri().scheme());
  ASSERT_EQ("/bar", builder.uri().path());
}

TEST(builder_test, assignment_operator_bug_116) {
  // https://github.com/cpp-netlib/uri/issues/116
  network::uri a("http://a.com:1234");
  ASSERT_TRUE(a.has_port());

  const network::uri b("http://b.com");
  ASSERT_FALSE(b.has_port());

  a = b;
  ASSERT_FALSE(a.has_port()) << a.string();
}

TEST(builder_test, construct_from_uri_bug_116) {
  // https://github.com/cpp-netlib/uri/issues/116
  network::uri a("http://a.com:1234");
  const network::uri b("http://b.com");
  a = b;

  network::uri_builder ub(a);
  const network::uri c(ub.uri());
  ASSERT_FALSE(c.has_port()) << c.string();
}

TEST(builder_test, append_query_value) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_component("q"));
  ASSERT_EQ(network::string_view("q"), ub.uri().query_begin()->first);
}

TEST(builder_test, append_query_value_encodes_equal_sign) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_component("="));
  ASSERT_EQ(network::string_view("%3D"), ub.uri().query_begin()->first);
}

TEST(builder_test, append_query_key_value_pair_encodes_equals_sign) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "="));
  ASSERT_EQ(network::string_view("q"), ub.uri().query_begin()->first);
  ASSERT_EQ(network::string_view("%3D"), ub.uri().query_begin()->second);
}

TEST(builder_test, append_query_key_value_pair_encodes_number_sign) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "#"));
  ASSERT_EQ(network::string_view("%23"), ub.uri().query_begin()->second);
}

TEST(builder_test, append_query_key_value_pair_encodes_percent_sign) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "%"));
  ASSERT_EQ(network::string_view("%25"), ub.uri().query_begin()->second);
}

TEST(builder_test, append_query_key_value_pair_encodes_ampersand) {
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "&"));
  ASSERT_EQ(network::string_view("%26"), ub.uri().query_begin()->second);
}

TEST(builder_test, append_query_key_value_pair_does_not_encode_slash) {
  // https://tools.ietf.org/html/rfc3986#section-3.4
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "/"));
  ASSERT_EQ(network::string_view("/"), ub.uri().query_begin()->second);
}

TEST(builder_test, append_query_key_value_pair_does_not_encode_qmark) {
  // https://tools.ietf.org/html/rfc3986#section-3.4
  network::uri_builder ub(network::uri("http://example.com"));
  ASSERT_NO_THROW(ub.append_query_key_value_pair("q", "?"));
  ASSERT_EQ(network::string_view("?"), ub.uri().query_begin()->second);
}

TEST(builder_test, build_from_uri_with_encoded_user_info) {
  network::uri_builder ub(network::uri("http://%40@example.com"));
  ASSERT_EQ(network::string_view("%40"), ub.uri().user_info());
}

TEST(builder_test, build_from_uri_with_encoded_query) {
  network::uri_builder ub(network::uri("http://example.com?x=%40"));
  ASSERT_EQ(network::string_view("x=%40"), ub.uri().query());
}

TEST(builder_test, build_from_uri_with_encoded_fragment) {
  network::uri_builder ub(network::uri("http://example.com#%40"));
  ASSERT_EQ(network::string_view("%40"), ub.uri().fragment());
}

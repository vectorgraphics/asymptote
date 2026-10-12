#pragma once

#include "LibLsp/JsonRpc/MessageIssue.h"
#include "LibLsp/JsonRpc/stream.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <rapidjson/document.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace test
{
inline int& Failures()
{
    static int failures = 0;
    return failures;
}

inline int& SkippedTests()
{
    static int skipped = 0;
    return skipped;
}

inline std::string& TestFilterStorage()
{
    static std::string pattern;
    return pattern;
}

inline void InitTestFilter(int argc, char** argv)
{
    std::string pattern;
    char const* env = std::getenv("LSPCPP_TEST_FILTER");
    if (env != nullptr)
    {
        pattern = env;
    }
    for (int i = 1; i < argc; ++i)
    {
        char const* arg = argv[i];
        char const* prefix = "--filter=";
        size_t const prefix_len = std::strlen(prefix);
        if (std::strncmp(arg, prefix, prefix_len) == 0)
        {
            pattern = arg + prefix_len;
            break;
        }
    }
    TestFilterStorage() = std::move(pattern);
}

inline bool ShouldRunTest(char const* test_name)
{
    std::string const& pattern = TestFilterStorage();
    if (pattern.empty())
    {
        return true;
    }
    return std::strstr(test_name, pattern.c_str()) != nullptr;
}

inline void Expect(bool condition, char const* message)
{
    if (!condition)
    {
        std::cerr << message << std::endl;
        ++Failures();
    }
}

class DummyLog : public lsp::Log
{
public:
    void log(Level, std::wstring&& msg) override
    {
        std::wcerr << msg << std::endl;
    }

    void log(Level, const std::wstring& msg) override
    {
        std::wcerr << msg << std::endl;
    }

    void log(Level, std::string&& msg) override
    {
        std::cerr << msg << std::endl;
    }

    void log(Level, const std::string& msg) override
    {
        std::cerr << msg << std::endl;
    }
};

class CollectingIssueHandler : public MessageIssueHandler
{
public:
    void handle(std::vector<MessageIssue>&&) override
    {
    }

    void handle(MessageIssue&& issue) override
    {
        issues.push_back(std::move(issue));
    }

    std::vector<MessageIssue> issues;
};

class StringIStream : public lsp::istream
{
public:
    explicit StringIStream(std::string data) : data_(std::move(data))
    {
    }

    void set_bad(bool value = true)
    {
        bad_ = value;
    }

    void set_fail(bool value = true)
    {
        fail_ = value;
    }

    int get() override
    {
        if (bad_)
        {
            return EOF;
        }
        if (pos_ >= data_.size())
        {
            eof_ = true;
            return EOF;
        }
        return static_cast<unsigned char>(data_[pos_++]);
    }

    lsp::istream& read(char* str, std::streamsize count) override
    {
        for (std::streamsize i = 0; i < count; ++i)
        {
            int const c = get();
            if (c == EOF)
            {
                break;
            }
            str[i] = static_cast<char>(c);
        }
        return *this;
    }

    bool fail() override
    {
        return fail_;
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return eof_;
    }

    bool good() override
    {
        return !fail_ && !bad_ && !eof_;
    }

    void clear() override
    {
        fail_ = false;
        bad_ = false;
        eof_ = false;
    }

    std::string what() override
    {
        return what_;
    }

private:
    std::string data_;
    size_t pos_ = 0;
    bool fail_ = false;
    bool bad_ = false;
    bool eof_ = false;
    std::string what_;
};

class FeedableIStream : public lsp::istream
{
public:
    void append(std::string const& data)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (char c : data)
            {
                data_.push_back(c);
            }
        }
        cv_.notify_all();
    }

    void interrupt() override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            interrupted_ = true;
        }
        cv_.notify_all();
    }

    int get() override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(
            lock,
            [&]()
            {
                return interrupted_ || !data_.empty();
            });
        if (data_.empty())
        {
            eof_ = true;
            return EOF;
        }
        auto const c = static_cast<unsigned char>(data_.front());
        data_.pop_front();
        return c;
    }

    lsp::istream& read(char* str, std::streamsize count) override
    {
        for (std::streamsize i = 0; i < count; ++i)
        {
            int const c = get();
            if (c == EOF)
            {
                break;
            }
            str[i] = static_cast<char>(c);
        }
        return *this;
    }

    bool fail() override
    {
        return false;
    }

    bool bad() override
    {
        return false;
    }

    bool eof() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return eof_;
    }

    bool good() override
    {
        return !eof();
    }

    void clear() override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        eof_ = false;
    }

    std::string what() override
    {
        return {};
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<char> data_;
    bool interrupted_ = false;
    bool eof_ = false;
};

class StringOStream : public lsp::ostream
{
public:
    void set_bad(bool value = true)
    {
        bad_ = value;
    }

    std::string const& data() const
    {
        return data_;
    }

    std::string snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_;
    }

    lsp::ostream& write(std::string const& value) override
    {
        if (!bad_)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data_ += value;
        }
        return *this;
    }

    lsp::ostream& write(std::streamsize value) override
    {
        return write(std::to_string(value));
    }

    lsp::ostream& flush() override
    {
        return *this;
    }

    bool fail() override
    {
        return fail_;
    }

    bool bad() override
    {
        return bad_;
    }

    bool eof() override
    {
        return false;
    }

    bool good() override
    {
        return !fail_ && !bad_;
    }

    void clear() override
    {
        fail_ = false;
        bad_ = false;
    }

    std::string what() override
    {
        return what_;
    }

private:
    mutable std::mutex mutex_;
    std::string data_;
    bool fail_ = false;
    bool bad_ = false;
    std::string what_;
};

inline std::string MakeLspFrame(std::string const& body)
{
    return "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
}

inline int ParseContentLength(std::string const& framed)
{
    auto const header_end = framed.find("\r\n\r\n");
    if (header_end == std::string::npos)
    {
        return -1;
    }
    auto const colon = framed.find(':');
    if (colon == std::string::npos)
    {
        return -1;
    }
    return std::atoi(framed.substr(colon + 1, header_end - colon - 1).c_str());
}

inline std::string WaitForOutputContaining(
    std::shared_ptr<StringOStream> const& output_stream,
    std::string const& needle,
    int attempts = 100,
    std::chrono::milliseconds delay = std::chrono::milliseconds(10))
{
    std::string output;
    for (int i = 0; i < attempts; ++i)
    {
        output = output_stream->snapshot();
        if (output.find(needle) != std::string::npos)
        {
            break;
        }
        std::this_thread::sleep_for(delay);
    }
    return output;
}

inline bool JsonEqual(
    rapidjson::Value const& lhs,
    rapidjson::Value const& rhs,
    std::string* diff_path = nullptr)
{
    if (lhs.GetType() != rhs.GetType())
    {
        if (diff_path != nullptr)
        {
            *diff_path = "type mismatch";
        }
        return false;
    }

    switch (lhs.GetType())
    {
    case rapidjson::kObjectType:
    {
        if (lhs.MemberCount() != rhs.MemberCount())
        {
            if (diff_path != nullptr)
            {
                *diff_path = "object member count mismatch";
            }
            return false;
        }
        for (auto const& member : lhs.GetObject())
        {
            auto it = rhs.FindMember(member.name);
            if (it == rhs.MemberEnd())
            {
                if (diff_path != nullptr)
                {
                    *diff_path = std::string("missing key: ") + member.name.GetString();
                }
                return false;
            }
            std::string child_path;
            if (!JsonEqual(member.value, it->value, diff_path != nullptr ? &child_path : nullptr))
            {
                if (diff_path != nullptr)
                {
                    *diff_path = std::string(member.name.GetString()) +
                        (child_path.empty() ? "" : "." + child_path);
                }
                return false;
            }
        }
        return true;
    }
    case rapidjson::kArrayType:
    {
        if (lhs.Size() != rhs.Size())
        {
            if (diff_path != nullptr)
            {
                *diff_path = "array length mismatch";
            }
            return false;
        }
        for (rapidjson::SizeType i = 0; i < lhs.Size(); ++i)
        {
            std::string child_path;
            if (!JsonEqual(lhs[i], rhs[i], diff_path != nullptr ? &child_path : nullptr))
            {
                if (diff_path != nullptr)
                {
                    *diff_path = std::to_string(i) + (child_path.empty() ? "" : "." + child_path);
                }
                return false;
            }
        }
        return true;
    }
    case rapidjson::kStringType:
        return std::strcmp(lhs.GetString(), rhs.GetString()) == 0;
    case rapidjson::kNumberType:
        return lhs.GetDouble() == rhs.GetDouble();
    case rapidjson::kTrueType:
    case rapidjson::kFalseType:
    case rapidjson::kNullType:
        return true;
    default:
        return false;
    }
}

inline bool JsonEqual(char const* actual_json, char const* expected_json, std::string* diff = nullptr)
{
    rapidjson::Document actual;
    rapidjson::Document expected;
    actual.Parse(actual_json);
    expected.Parse(expected_json);
    if (actual.HasParseError() || expected.HasParseError())
    {
        if (diff != nullptr)
        {
            *diff = "json parse error";
        }
        return false;
    }
    return JsonEqual(actual, expected, diff);
}

inline void ExpectJsonEqual(char const* actual, char const* expected, char const* message)
{
    std::string diff;
    if (!JsonEqual(actual, expected, &diff))
    {
        std::cerr << message;
        if (!diff.empty())
        {
            std::cerr << " (first diff at: " << diff << ")";
        }
        std::cerr << "\nexpected: " << expected << "\nactual:   " << actual << std::endl;
        ++Failures();
    }
}

inline void ExpectJsonEqual(std::string const& actual, std::string const& expected, char const* message)
{
    ExpectJsonEqual(actual.c_str(), expected.c_str(), message);
}

inline bool UpdateFixturesEnabled()
{
    char const* env = std::getenv("LSPCPP_UPDATE_FIXTURES");
    return env != nullptr && env[0] != '\0';
}

inline std::string ReadTextFile(std::vector<std::string> const& candidates)
{
    for (auto const& candidate : candidates)
    {
        std::ifstream input(candidate.c_str());
        if (!input)
        {
            continue;
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }
    return {};
}

inline std::vector<std::string> FixturePathCandidates(char const* name)
{
    std::string const suffix = std::string("tests/fixtures/lsp/") + name;
    return {
        suffix,
        "../" + suffix,
        "../../" + suffix,
        "../../../" + suffix,
    };
}

inline std::string ReadFixture(char const* name)
{
    std::string const json = ReadTextFile(FixturePathCandidates(name));
    Expect(!json.empty(), "protocol golden fixture must be readable");
    return json;
}

inline bool WriteFixture(char const* name, std::string const& content)
{
    if (!UpdateFixturesEnabled())
    {
        return false;
    }
    for (auto const& candidate : FixturePathCandidates(name))
    {
        std::ofstream output(candidate.c_str());
        if (!output)
        {
            continue;
        }
        output << content;
        return true;
    }
    std::string const primary = std::string("tests/fixtures/lsp/") + name;
    std::ofstream output(primary.c_str());
    if (!output)
    {
        return false;
    }
    output << content;
    return true;
}

inline void ExpectJsonFixture(std::string const& actual, char const* fixture_name, char const* message)
{
    if (UpdateFixturesEnabled())
    {
        WriteFixture(fixture_name, actual);
        return;
    }
    ExpectJsonEqual(actual, ReadFixture(fixture_name), message);
}

inline std::vector<std::string> ExtractLspFrameBodies(std::string const& output)
{
    std::vector<std::string> bodies;
    size_t pos = 0;
    while (pos < output.size())
    {
        size_t const header_end = output.find("\r\n\r\n", pos);
        if (header_end == std::string::npos)
        {
            break;
        }
        size_t const length_start = output.find("Content-Length:", pos);
        if (length_start == std::string::npos || length_start > header_end)
        {
            break;
        }
        size_t const value_start = length_start + std::string("Content-Length:").size();
        int const length = std::atoi(output.substr(value_start, header_end - value_start).c_str());
        size_t const body_start = header_end + 4;
        if (length < 0 || body_start + static_cast<size_t>(length) > output.size())
        {
            break;
        }
        bodies.push_back(output.substr(body_start, static_cast<size_t>(length)));
        pos = body_start + static_cast<size_t>(length);
    }
    return bodies;
}

} // namespace test

#define RUN_TEST(fn) \
    do { \
        if (test::ShouldRunTest(#fn)) { \
            (fn)(); \
        } else { \
            ++test::SkippedTests(); \
        } \
    } while (0)


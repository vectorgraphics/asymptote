
#include "LibLsp/JsonRpc/StreamMessageProducer.h"
#include <cassert>
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>

#include "LibLsp/JsonRpc/stream.h"
#include "LibLsp/lsp/Markup/string_ref.h"

bool StartsWith(std::string value, std::string start);
bool StartsWith(std::string value, std::string start)
{
    if (start.size() > value.size())
    {
        return false;
    }
    return std::equal(start.begin(), start.end(), value.begin());
}

using namespace std;
namespace
{
string JSONRPC_VERSION = "2.0";
string CONTENT_LENGTH_HEADER = "Content-Length";
string CONTENT_TYPE_HEADER = "Content-Type";
string JSON_MIME_TYPE = "application/json";
string CRLF = "\r\n";

bool parseContentLength(std::string const& value, int& contentLength)
{
    errno = 0;
    char* end = nullptr;
    long const parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || errno == ERANGE || parsed < 0 || parsed > INT_MAX)
    {
        return false;
    }
    while (*end == ' ' || *end == '\t')
    {
        ++end;
    }
    if (*end != '\0')
    {
        return false;
    }
    contentLength = static_cast<int>(parsed);
    return true;
}

bool reportInputState(std::shared_ptr<lsp::istream> const& input, MessageIssueHandler& issueHandler)
{
    if (input->bad())
    {
        std::string info = "Input stream is bad.";
        auto what = input->what();
        if (what.size())
        {
            info += "Reason:";
            info += input->what();
        }
        MessageIssue issue(info, lsp::Log::Level::SEVERE);
        issueHandler.handle(std::move(issue));
        return false;
    }
    if (input->fail())
    {
        std::string info = "Input fail.";
        auto what = input->what();
        if (what.size())
        {
            info += "Reason:";
            info += input->what();
        }
        MessageIssue issue(info, lsp::Log::Level::WARNING);
        issueHandler.handle(std::move(issue));
        if (input->need_to_clear_the_state())
        {
            input->clear();
            return true;
        }
        return false;
    }
    return true;
}

bool readMore(std::shared_ptr<lsp::istream> const& input, std::string& buffer)
{
    std::array<char, 4096> chunk {};
    auto const read = input->read_some(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    if (read <= 0)
    {
        return false;
    }
    buffer.append(chunk.data(), static_cast<size_t>(read));
    return true;
}

bool discardBytes(
    std::shared_ptr<lsp::istream> const& input,
    size_t count,
    MessageIssueHandler& issueHandler
)
{
    std::array<char, 4096> chunk {};
    while (count != 0)
    {
        auto const toRead = std::min(count, chunk.size());
        input->read(chunk.data(), static_cast<std::streamsize>(toRead));
        if (input->bad())
        {
            MessageIssue issue("Input stream is bad while discarding oversized frame.", lsp::Log::Level::SEVERE);
            issueHandler.handle(std::move(issue));
            return false;
        }
        if (input->eof() || input->fail())
        {
            MessageIssue issue("Input ended while discarding oversized frame.", lsp::Log::Level::WARNING);
            issueHandler.handle(std::move(issue));
            return false;
        }
        count -= toRead;
    }
    return true;
}

std::string::size_type findHeaderEnd(std::string const& buffer, size_t& delimiterSize)
{
    auto const crlf = buffer.find("\r\n\r\n");
    auto const lf = buffer.find("\n\n");
    if (crlf == std::string::npos)
    {
        if (lf == std::string::npos)
        {
            return std::string::npos;
        }
        delimiterSize = 2;
        return lf;
    }
    if (lf == std::string::npos || crlf < lf)
    {
        delimiterSize = 4;
        return crlf;
    }
    delimiterSize = 2;
    return lf;
}

void parseHeaderBlock(std::string const& block, LSPStreamMessageProducer& producer, LSPStreamMessageProducer::Headers& headers)
{
    size_t start = 0;
    while (start <= block.size())
    {
        auto end = block.find('\n', start);
        if (end == std::string::npos)
        {
            end = block.size();
        }
        auto line = block.substr(start, end - start);
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            producer.parseHeader(line, headers);
        }
        if (end == block.size())
        {
            break;
        }
        start = end + 1;
    }
}

} // namespace

bool StreamMessageProducer::reportOverload(std::string const& message)
{
    MessageIssue issue(message, lsp::Log::Level::WARNING);
    issueHandler.handle(std::move(issue));
    if (!overloadHandler)
    {
        return false;
    }
    return overloadHandler(message);
}

void LSPStreamMessageProducer::parseHeader(std::string& line, LSPStreamMessageProducer::Headers& headers)
{
    std::string::size_type const sepIndex = line.find(':');
    if (sepIndex != std::string::npos)
    {
        auto key = line.substr(0, sepIndex);
        if (key == CONTENT_LENGTH_HEADER)
        {
            int contentLength = -1;
            if (parseContentLength(line.substr(sepIndex + 1), contentLength))
            {
                headers.contentLength = contentLength;
            }
        }
        else if (key == CONTENT_TYPE_HEADER)
        {
            std::string::size_type const charsetIndex = line.find("charset=");
            if (charsetIndex != std::string::npos)
            {
                headers.charset = line.substr(charsetIndex + 8);
            }
        }
    }
}

void LSPStreamMessageProducer::listen(MessageConsumer callBack)
{
    if (!input)
    {
        return;
    }

    keepRunning.store(true, std::memory_order_relaxed);
    std::string buffer;
    while (keepRunning.load(std::memory_order_relaxed))
    {
        if (!reportInputState(input, issueHandler))
        {
            return;
        }

        size_t delimiterSize = 0;
        auto headerEnd = findHeaderEnd(buffer, delimiterSize);
        while (headerEnd == std::string::npos && keepRunning.load(std::memory_order_relaxed))
        {
            if (!readMore(input, buffer))
            {
                keepRunning.store(false, std::memory_order_relaxed);
                if (!buffer.empty())
                {
                    MessageIssue issue("No more input when reading message headers", lsp::Log::Level::INFO);
                    issueHandler.handle(std::move(issue));
                }
                return;
            }
            if (!reportInputState(input, issueHandler))
            {
                return;
            }
            headerEnd = findHeaderEnd(buffer, delimiterSize);
        }
        if (!keepRunning.load(std::memory_order_relaxed))
        {
            return;
        }

        Headers headers;
        auto headerBlock = buffer.substr(0, headerEnd);
        parseHeaderBlock(headerBlock, *this, headers);
        buffer.erase(0, headerEnd + delimiterSize);

        if (headers.contentLength < 0)
        {
            string info = "Unexpected token:" + headerBlock;
            info += "  (expected Content-Length: sequence);";
            MessageIssue issue(info, lsp::Log::Level::WARNING);
            issueHandler.handle(std::move(issue));
            continue;
        }

        auto const contentLength = static_cast<size_t>(headers.contentLength);
        if (maxFrameSize != 0 && contentLength > maxFrameSize)
        {
            std::string info = "LSP frame size ";
            info += std::to_string(contentLength);
            info += " exceeds configured maximum ";
            info += std::to_string(maxFrameSize);
            info += ".";
            if (!reportOverload(info))
            {
                keepRunning.store(false, std::memory_order_relaxed);
                return;
            }

            auto const buffered = std::min(buffer.size(), contentLength);
            buffer.erase(0, buffered);
            auto const remaining = contentLength - buffered;
            if (remaining != 0 && !discardBytes(input, remaining, issueHandler))
            {
                keepRunning.store(false, std::memory_order_relaxed);
                return;
            }
            continue;
        }
        if (buffer.size() < contentLength)
        {
            // Bulk-read the remaining body with a single istream::read() call.
            // Header scanning uses read_some (fast for standard_istream, and a
            // safe byte-by-byte fallback for custom streams), but the body length
            // is known here, so this preserves the historical bulk-read behavior
            // for custom istream implementations that only override read().
            size_t const already = buffer.size();
            size_t const needed = contentLength - already;
            buffer.resize(contentLength);
            input->read(&buffer[already], static_cast<std::streamsize>(needed));

            if (input->bad())
            {
                std::string info = "Input stream is bad.";
                auto what = input->what();
                if (!what.empty())
                {
                    info += "Reason:";
                    info += what;
                }
                MessageIssue issue(info, lsp::Log::Level::SEVERE);
                issueHandler.handle(std::move(issue));
                keepRunning.store(false, std::memory_order_relaxed);
                return;
            }
            if (input->eof())
            {
                MessageIssue issue("No more input when reading content body", lsp::Log::Level::INFO);
                issueHandler.handle(std::move(issue));
                keepRunning.store(false, std::memory_order_relaxed);
                return;
            }
            if (input->fail())
            {
                std::string info = "Input fail.";
                auto what = input->what();
                if (!what.empty())
                {
                    info += "Reason:";
                    info += what;
                }
                MessageIssue issue(info, lsp::Log::Level::WARNING);
                issueHandler.handle(std::move(issue));
                keepRunning.store(false, std::memory_order_relaxed);
                return;
            }
        }

        std::string content = buffer.substr(0, contentLength);
        buffer.erase(0, contentLength);
        callBack(std::move(content));
        if (input->eof() && buffer.empty())
        {
            keepRunning.store(false, std::memory_order_relaxed);
        }
    }
}

void LSPStreamMessageProducer::bind(std::shared_ptr<lsp::istream> _in)
{
    input = _in;
}

bool LSPStreamMessageProducer::handleMessage(Headers& headers, MessageConsumer callBack)
{
    // Read content.
    auto content_length = headers.contentLength;
    std::string content(content_length, 0);
    if (content_length > 0)
    {
        auto data = &content[0];
        input->read(data, content_length);
    }
    if (input->bad())
    {
        std::string info = "Input stream is bad.";
        auto what = input->what();
        if (!what.empty())
        {
            info += "Reason:";
            info += input->what();
        }
        MessageIssue issue(info, lsp::Log::Level::SEVERE);
        issueHandler.handle(std::move(issue));
        return false;
    }

    if (input->eof())
    {
        MessageIssue issue("No more input when reading content body", lsp::Log::Level::INFO);
        issueHandler.handle(std::move(issue));
        return false;
    }
    if (input->fail())
    {
        std::string info = "Input fail.";
        auto what = input->what();
        if (!what.empty())
        {
            info += "Reason:";
            info += input->what();
        }
        MessageIssue issue(info, lsp::Log::Level::WARNING);
        issueHandler.handle(std::move(issue));
        return false;
    }

    callBack(std::move(content));

    return true;
}

/// For lit tests we support a simplified syntax:
/// - messages are delimited by '// -----' on a line by itself
/// - lines starting with // are ignored.
/// This is a testing path, so favor simplicity over performance here.

void DelimitedStreamMessageProducer::listen(MessageConsumer callBack)
{
    if (!input)
    {
        return;
    }

    keepRunning.store(true, std::memory_order_relaxed);

    auto readLine = [&](std::string_ref& lineBuilder) -> bool
    {
        while (keepRunning.load(std::memory_order_relaxed))
        {
            if (input->bad())
            {
                std::string info = "Input stream is bad.";
                auto what = input->what();
                if (what.size())
                {
                    info += "Reason:";
                    info += input->what();
                }
                MessageIssue issue(info, lsp::Log::Level::SEVERE);
                issueHandler.handle(std::move(issue));
                return false;
            }
            if (input->fail())
            {
                std::string info = "Input fail.";
                auto what = input->what();
                if (what.size())
                {
                    info += "Reason:";
                    info += input->what();
                }
                MessageIssue issue(info, lsp::Log::Level::WARNING);
                issueHandler.handle(std::move(issue));
                if (input->need_to_clear_the_state())
                {
                    input->clear();
                }
                else
                {
                    return false;
                }
            }
            int c = input->get();
            if (c == EOF)
            {
                // End of input stream has been reached
                keepRunning.store(false, std::memory_order_relaxed);
            }
            else
            {
                if (c == '\n')
                {
                    if (!lineBuilder.empty())
                    {
                        lineBuilder.push_back(static_cast<char>(c));
                        return true;
                    }
                }
                else if (c != '\r')
                {
                    // Add the input to the current header line

                    lineBuilder.push_back((char)c);
                }
            }
        }
        return false;
    };

    auto getMessage = [&](std::string& json) -> bool
    {
        std::string_ref lineBuilder;
        auto trimWhitespace = [](std::string_ref& line)
        {
            while (!line.empty() && (line.front() == ' ' || line.front() == '\t' || line.front() == '\r' ||
                                     line.front() == '\n'))
            {
                line.erase(line.begin());
            }
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r' ||
                                     line.back() == '\n'))
            {
                line.pop_back();
            }
        };
        while (readLine(lineBuilder))
        {
            trimWhitespace(lineBuilder);
            if (lineBuilder.start_with("//"))
            {
                // Found a delimiter for the message.
                if (lineBuilder == "// -----")
                {
                    return true;
                }
            }
            else
            {
                json += lineBuilder;
                if (maxFrameSize != 0 && json.size() > maxFrameSize)
                {
                    std::string info = "Delimited frame size exceeds configured maximum ";
                    info += std::to_string(maxFrameSize);
                    info += ".";
                    if (!reportOverload(info))
                    {
                        keepRunning.store(false, std::memory_order_relaxed);
                        return false;
                    }
                    json.clear();
                }
            }
            lineBuilder.clear();
        }
        return false;
    };

    while (true)
    {
        std::string json;
        if (getMessage(json))
        {
            callBack(std::move(json));
        }
        else
        {
            return;
        }
    }
}

void DelimitedStreamMessageProducer::bind(std::shared_ptr<lsp::istream> _in)
{
    input = _in;
}

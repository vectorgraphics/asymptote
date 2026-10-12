#pragma once

#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "test_helpers.h"

#include <chrono>
#include <memory>
#include <rapidjson/document.h>
#include <string>
#include <thread>

namespace test
{

// Lightweight client-side test double for asserting server-initiated RPC traffic.
class LSPClient
{
public:
    void attach(
        std::shared_ptr<StringOStream> const& server_output,
        std::shared_ptr<FeedableIStream> const& server_input
    )
    {
        output_ = server_output;
        input_ = server_input;
    }

    bool expectServerCall(char const* method, int attempts = 100)
    {
        for (int i = 0; i < attempts; ++i)
        {
            if (output_ && output_->snapshot().find(method) != std::string::npos)
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return output_ && output_->snapshot().find(method) != std::string::npos;
    }

    std::string takeLastServerCallBody(char const* method) const
    {
        if (!output_)
        {
            return {};
        }
        auto const bodies = ExtractLspFrameBodies(output_->snapshot());
        for (auto it = bodies.rbegin(); it != bodies.rend(); ++it)
        {
            if (it->find(method) != std::string::npos)
            {
                return *it;
            }
        }
        return {};
    }

    rapidjson::Document takeCallParams(char const* method) const
    {
        rapidjson::Document params;
        params.SetNull();
        std::string const body = takeLastServerCallBody(method);
        if (body.empty())
        {
            return params;
        }

        rapidjson::Document message;
        message.Parse(body.c_str());
        if (message.HasParseError() || !message.IsObject() || !message.HasMember("params"))
        {
            return params;
        }
        params.CopyFrom(message["params"], params.GetAllocator());
        return params;
    }

    void reply(std::string const& body) const
    {
        if (input_)
        {
            input_->append(MakeLspFrame(body));
        }
    }

private:
    std::shared_ptr<StringOStream> output_;
    std::shared_ptr<FeedableIStream> input_;
};

} // namespace test

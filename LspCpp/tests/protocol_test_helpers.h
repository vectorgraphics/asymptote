#pragma once

#include "LibLsp/JsonRpc/json.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"

#include "test_helpers.h"

#include <memory>
#include <rapidjson/document.h>

namespace test
{

inline JsonReader MakeJsonReader(rapidjson::Document& document, char const* json)
{
    document.Parse(json);
    return JsonReader(&document);
}

inline std::unique_ptr<LspMessage> ParseProtocolRequest(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json)
{
    rapidjson::Document document;
    JsonReader reader = MakeJsonReader(document, json);
    return handler.parseRequstMessage(method, reader);
}

inline std::unique_ptr<LspMessage> ParseProtocolNotification(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json)
{
    rapidjson::Document document;
    JsonReader reader = MakeJsonReader(document, json);
    return handler.parseNotificationMessage(method, reader);
}

inline std::unique_ptr<LspMessage> ParseProtocolResponse(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json)
{
    rapidjson::Document document;
    JsonReader reader = MakeJsonReader(document, json);
    return handler.parseResponseMessage(method, reader);
}

inline void ExpectParsesRequest(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json, char const* message)
{
    Expect(ParseProtocolRequest(handler, method, json) != nullptr, message);
}

inline void ExpectParsesNotification(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json, char const* message)
{
    Expect(ParseProtocolNotification(handler, method, json) != nullptr, message);
}

inline void ExpectParsesResponse(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json, char const* message)
{
    Expect(ParseProtocolResponse(handler, method, json) != nullptr, message);
}

inline void ExpectParsesErrorResponse(
    lsp::ProtocolJsonHandler& handler, MethodType method, char const* json, char const* message)
{
    std::unique_ptr<LspMessage> msg = ParseProtocolResponse(handler, method, json);
    Expect(msg != nullptr, message);
    auto* error = dynamic_cast<Rsp_Error*>(msg.get());
    Expect(error != nullptr, "error response must deserialize as Rsp_Error");
    if (error != nullptr)
    {
        Expect(error->error.message == "Request failed", "error response message must round-trip");
        Expect(
            error->error.code == lsErrorCodes::InternalError,
            "error response code must round-trip as InternalError");
    }
}

} // namespace test

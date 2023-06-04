#include "LibLsp/lsp/lsp_diagnostic.h"

bool lsDiagnostic::operator==(const lsDiagnostic& rhs) const {
  // Just check the important fields.
  return range == rhs.range && message == rhs.message;
}
bool lsDiagnostic::operator!=(const lsDiagnostic& rhs) const {
  return !(*this == rhs);
}

std::string lsResponseError::ToString()
{
        std::string info = "code:";
        switch (code)
        {
        case lsErrorCodes::ParseError:
                info += "ParseError\n";
                break;
        case lsErrorCodes::InvalidRequest:
                info += "InvalidRequest\n";
                break;
        case lsErrorCodes::MethodNotFound:
                info += "MethodNotFound\n";
                break;
        case lsErrorCodes::InvalidParams:
                info += "InvalidParams\n";
                break;
        case lsErrorCodes::InternalError:
                info += "InternalError\n";
                break;
        case lsErrorCodes::serverErrorStart:
                info += "serverErrorStart\n";
                break;
        case lsErrorCodes::serverErrorEnd:
                info += "serverErrorEnd\n";
                break;
        case lsErrorCodes::ServerNotInitialized:
                info += "ServerNotInitialized\n";
                break;
        case lsErrorCodes::UnknownErrorCode:
                info += "UnknownErrorCode\n";
                break;
                // Defined by the protocol.
        case lsErrorCodes::RequestCancelled:
                info += "RequestCancelled\n";
                break;
        default:
                {
                        std::stringstream ss;
                        ss << "unknown code:" << (int32_t)code << std::endl;
                        info += ss.str();
                }
                break;
        }
        info += "message:" + message;
        info += "\n";

        if(data.has_value())
        {

                info += "data:" + data.value().Data();
                info += "\n";
        }
        return info;
}

void lsResponseError::Write(Writer& visitor) {
        auto& value = *this;
        int code2 = static_cast<int>(this->code);

        visitor.StartObject();
        REFLECT_MEMBER2("code", code2);
        REFLECT_MEMBER(message);
        visitor.EndObject();
}

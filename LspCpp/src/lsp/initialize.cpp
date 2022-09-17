#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/JsonRpc/json.h"

void Reflect(Reader& reader, lsInitializeParams::lsTrace& value)
{
        if (!reader.IsString())
        {
                value = lsInitializeParams::lsTrace::Off;
                return;
        }
        std::string v = reader.GetString();
        if (v == "off")
                value = lsInitializeParams::lsTrace::Off;
        else if (v == "messages")
                value = lsInitializeParams::lsTrace::Messages;
        else if (v == "verbose")
                value = lsInitializeParams::lsTrace::Verbose;
}

void Reflect(Writer& writer, lsInitializeParams::lsTrace& value)
{
        switch (value)
        {
        case lsInitializeParams::lsTrace::Off:
                writer.String("off");
                break;
        case lsInitializeParams::lsTrace::Messages:
                writer.String("messages");
                break;
        case lsInitializeParams::lsTrace::Verbose:
                writer.String("verbose");
                break;
        }
}
 void Reflect(Reader& visitor, std::pair<boost::optional<lsTextDocumentSyncKind>, boost::optional<lsTextDocumentSyncOptions> >& value)
{
        if(((JsonReader&)visitor).m_->IsObject())
        {
                Reflect(visitor, value.second);
        }
        else
        {
                Reflect(visitor, value.first);
        }
}

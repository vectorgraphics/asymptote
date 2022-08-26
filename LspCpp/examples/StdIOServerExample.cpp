
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include <boost/program_options.hpp>
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/textDocument/typeHierarchy.h"
#include "LibLsp/lsp/AbsolutePath.h"
#include "LibLsp/lsp/textDocument/resolveCompletionItem.h"
#include <network/uri.hpp>

#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/stream.h"
#include "LibLsp/JsonRpc/TcpServer.h"
#include "LibLsp/lsp/textDocument/document_symbol.h"
#include "LibLsp/lsp/workspace/execute_command.h"
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include <boost/asio.hpp>
#include <iostream>
using namespace boost::asio::ip;
using namespace std;
class DummyLog :public lsp::Log
{
public:

        void log(Level level, std::wstring&& msg)
        {

                std::wcerr << msg << std::endl;
        };
        void log(Level level, const std::wstring& msg)
        {
                std::wcerr << msg << std::endl;
        };
        void log(Level level, std::string&& msg)
        {
                std::cerr << msg << std::endl;
        };
        void log(Level level, const std::string& msg)
        {
                std::cerr << msg << std::endl;
        };
};


class StdIOServer
{
public:

        StdIOServer() : remote_end_point_(protocol_json_handler, endpoint, _log)
        {
                remote_end_point_.registerHandler([&](const td_initialize::request& req)
                {
                                td_initialize::response rsp;
                                rsp.id = req.id;
                                CodeLensOptions code_lens_options;
                                code_lens_options.resolveProvider = true;
                                rsp.result.capabilities.codeLensProvider = code_lens_options;
                                return rsp;
                });

                remote_end_point_.registerHandler([&](Notify_Exit::notify& notify)
                        {
                remote_end_point_.stop();
                                esc_event.notify(std::make_unique<bool>(true));
                        });
                remote_end_point_.registerHandler([&](const td_definition::request& req,
                        const CancelMonitor& monitor)
                        {
                                td_definition::response rsp;
                                rsp.result.first = std::vector<lsLocation>();
                                if (monitor && monitor())
                                {
                                        _log.info("textDocument/definition request had been cancel.");
                                }
                                return rsp;
                        });

                remote_end_point_.startProcessingMessages(input, output);
        }
        ~StdIOServer()
        {

        }

        struct ostream : lsp::base_ostream<std::ostream>
        {
                explicit ostream(std::ostream& _t)
                        : base_ostream<std::ostream>(_t)
                {

                }

                std::string what() override
                {
                        return {};
                }
        };
        struct istream :lsp::base_istream<std::istream>
        {
                explicit istream(std::istream& _t)
                        : base_istream<std::istream>(_t)
                {
                }

                std::string what() override
                {
                        return {};
                }
        };
        std::shared_ptr < lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared < lsp::ProtocolJsonHandler >();
        DummyLog _log;

        std::shared_ptr<ostream> output = std::make_shared<ostream>(std::cout);
        std::shared_ptr<istream> input = std::make_shared<istream>(std::cin);

        std::shared_ptr < GenericEndpoint >  endpoint = std::make_shared<GenericEndpoint>(_log);
        RemoteEndPoint remote_end_point_;
        Condition<bool> esc_event;
};




int main(int argc, char* argv[])
{
        using namespace  boost::program_options;
        options_description desc("Allowed options");
        desc.add_options()
                ("help,h", "produce help message");



        variables_map vm;
        try {
                store(parse_command_line(argc, argv, desc), vm);
        }
        catch (std::exception& e) {
                std::cout << "Undefined input.Reason:" << e.what() << std::endl;
                return 0;
        }
        notify(vm);


        if (vm.count("help"))
        {
                cout << desc << endl;
                return 1;
        }
        StdIOServer server;
        server.esc_event.wait();

        return 0;
}



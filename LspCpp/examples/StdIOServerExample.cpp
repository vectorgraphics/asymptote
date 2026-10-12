
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include <boost/program_options.hpp>
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"

#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/stream.h"
#include <cstdlib>
#include <iostream>

using namespace std;

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

        std::shared_ptr < lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared < lsp::ProtocolJsonHandler >();
        lsp::StderrLog _log;

        std::shared_ptr<lsp::ostream> output = lsp::make_stdout_stream();
        std::shared_ptr<lsp::istream> input = lsp::make_stdin_stream();

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

        std::_Exit(0);
}



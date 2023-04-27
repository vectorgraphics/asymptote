
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
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
                std::wcout << msg << std::endl;
        };
        void log(Level level, const std::wstring& msg)
        {
                std::wcout << msg << std::endl;
        };
        void log(Level level, std::string&& msg)
        {
                std::cout << msg << std::endl;
        };
        void log(Level level, const std::string& msg)
        {
                std::cout << msg << std::endl;
        };
};

std::string _address = "127.0.0.1";
std::string _port = "9333";

class Server
{
public:


        Server():server(_address,_port,protocol_json_handler, endpoint, _log)
        {
                server.point.registerHandler(
                        [&](const td_initialize::request& req)
                          ->lsp::ResponseOrError< td_initialize::response >{

                                td_initialize::response rsp;
                                CodeLensOptions code_lens_options;
                                code_lens_options.resolveProvider = true;
                                rsp.result.capabilities.codeLensProvider = code_lens_options;

                                return rsp;
                        });
                server.point.registerHandler([&](const td_definition::request& req
                        ,const CancelMonitor& monitor)  -> lsp::ResponseOrError<td_definition::response>
                        {

                                std::this_thread::sleep_for(std::chrono::seconds(8));

                            if( monitor && monitor() )
                            {
                                        _log.info("textDocument/definition request had been cancel.");
                    Rsp_Error rsp;
                    rsp.error.code = lsErrorCodes::RequestCancelled;
                    rsp.error.message = "textDocument/definition request had been cancel.";
                    return  rsp;
                            }
                else
                {
                    td_definition::response rsp;
                    rsp.result.first= std::vector<lsLocation>();
                    return rsp;
                }

                        });

                server.point.registerHandler([=](Notify_Exit::notify& notify)
                        {
                                std::cout << notify.ToJson() << std::endl;
                        });
                std::thread([&]()
                        {
                                server.run();
                        }).detach();
        }
        ~Server()
        {
                server.stop();
        }
        std::shared_ptr < lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared < lsp::ProtocolJsonHandler >();
        DummyLog _log;

        std::shared_ptr < GenericEndpoint >  endpoint = std::make_shared<GenericEndpoint>(_log);
        lsp::TcpServer server;

};

class Client
{
public:
        struct iostream :public lsp::base_iostream<tcp::iostream>
        {
                explicit iostream(boost::asio::basic_socket_iostream<tcp>& _t)
                        : base_iostream<boost::asio::basic_socket_iostream<tcp>>(_t)
                {
                }

                std::string what() override
                {
                        auto  msg = _impl.error().message();
                        return  msg;
                }

        };
        Client() :remote_end_point_(protocol_json_handler, endpoint, _log)
        {
                tcp::endpoint end_point( address::from_string(_address), 9333);

                socket_ = std::make_shared<tcp::iostream>();
                socket_->connect(end_point);
                if (!socket_)
                {
                        string temp = "Unable to connect: " + socket_->error().message();
                        std::cout << temp << std::endl;
                }

                vector<string> args;
                socket_proxy = std::make_shared<iostream>(*socket_.get());

                remote_end_point_.startProcessingMessages(socket_proxy, socket_proxy);
        }
        ~Client()
        {
        remote_end_point_.stop();
                std::this_thread::sleep_for(std::chrono::milliseconds (1000));
                socket_->close();
        }

        std::shared_ptr < lsp::ProtocolJsonHandler >  protocol_json_handler = std::make_shared< lsp::ProtocolJsonHandler>();
        DummyLog _log;

        std::shared_ptr<GenericEndpoint>  endpoint = std::make_shared<GenericEndpoint>(_log);

        std::shared_ptr < iostream> socket_proxy;
        std::shared_ptr<tcp::iostream>  socket_;
        RemoteEndPoint remote_end_point_;
};

int main()
{

        Server server;
        Client client;

        {
                td_initialize::request req;
                auto rsp = client.remote_end_point_.waitResponse(req);
                if (rsp)
                {
                        std::cout << rsp->response.ToJson() << std::endl;
                }
                else
                {
                        std::cout << "get initialze  response time out" << std::endl;
                }
        }
        {
                td_definition::request req;
                auto future_rsp = client.remote_end_point_.send(req);
        client.remote_end_point_.cancelRequest(req.id);

                auto state = future_rsp.wait_for(std::chrono::seconds(16));
                if (lsp::future_status::timeout == state)
                {
                        std::cout << "get textDocument/definition  response time out" << std::endl;
                        return 0;
                }
                auto rsp = future_rsp.get();
                if (rsp.is_error)
                {
                        std::cout << "get textDocument/definition  response error :" << rsp.ToJson() << std::endl;

                }
                else
                {
                        std::cout << rsp.response.ToJson() << std::endl;
                }
        }
        Notify_Exit::notify notify;
        client.remote_end_point_.send(notify);
        return 0;
}



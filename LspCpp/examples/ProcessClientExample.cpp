#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/Process.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialize.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
class ExampleLog : public lsp::Log
{
public:
    void log(Level, std::wstring&& msg) override
    {
        std::wcerr << msg << std::endl;
    }
    void log(Level, std::wstring const& msg) override
    {
        std::wcerr << msg << std::endl;
    }
    void log(Level, std::string&& msg) override
    {
        std::cerr << msg << std::endl;
    }
    void log(Level, std::string const& msg) override
    {
        std::cerr << msg << std::endl;
    }
};

std::string ParseExecPath(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string const arg = argv[i];
        if (arg == "--execPath" && i + 1 < argc)
        {
            return argv[i + 1];
        }
        if (arg.rfind("--execPath=", 0) == 0)
        {
            return arg.substr(std::string("--execPath=").size());
        }
    }
    return "StdIOServerExample";
}

std::vector<std::string> ParseServerArgs(int argc, char* argv[])
{
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
    {
        std::string const arg = argv[i];
        if (arg == "--execPath")
        {
            ++i;
            continue;
        }
        if (arg.rfind("--execPath=", 0) == 0)
        {
            continue;
        }
        args.push_back(arg);
    }
    return args;
}
} // namespace

int main(int argc, char* argv[])
{
    std::string const exec_path = ParseExecPath(argc, argv);
    std::vector<std::string> const server_args = ParseServerArgs(argc, argv);

    ExampleLog log;
    auto json_handler = std::make_shared<lsp::ProtocolJsonHandler>();
    auto endpoint = std::make_shared<GenericEndpoint>(log);

    try
    {
        lsp::Process process = lsp::Process::start(exec_path, server_args);
        RemoteEndPoint point(json_handler, endpoint, log);
        point.startProcessingMessages(process.input(), process.output());

        {
            td_initialize::request req;
            auto rsp = point.waitResponse(req, 5000);
            if (!rsp)
            {
                std::cerr << "initialize response timed out" << std::endl;
                point.stop();
                process.terminate();
                return 1;
            }
            std::cerr << rsp->ToJson() << std::endl;
        }

        Notify_Exit::notify notify;
        point.send(notify);
        point.stop();

        for (int i = 0; i < 100 && process.isRunning(); ++i)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (process.isRunning())
        {
            process.terminate();
        }
        process.wait();
    }
    catch (lsp::ProcessError const& error)
    {
        std::cerr << "Failed to start language server: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}

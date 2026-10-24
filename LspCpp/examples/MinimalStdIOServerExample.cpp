#include "LibLsp/LspCpp.h"

int main()
{
    lsp::LanguageSession server;
    Condition<bool> exit_requested;

    server.on(
        [](td_initialize::request const& req)
        {
            td_initialize::response rsp;
            rsp.id = req.id;
            return rsp;
        });

    server.on(
        [&](Notify_Exit::notify&)
        {
            exit_requested.notify(std::make_unique<bool>(true));
        });

    server.startStdio();
    exit_requested.wait();
    server.stop();
    return 0;
}

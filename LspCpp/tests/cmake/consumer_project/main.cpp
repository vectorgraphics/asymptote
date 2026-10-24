#include "LibLsp/LspCpp.h"

int main()
{
    lsp::LanguageSession session;
    return session.protocolJsonHandler() && session.localEndpoint() ? 0 : 1;
}

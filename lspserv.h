#pragma once
#include "common.h"
#include "symbolmaps.h"

#include <LibLsp/lsp/ProtocolJsonHandler.h>
#include <LibLsp/lsp/AbsolutePath.h>

#include <LibLsp/JsonRpc/Endpoint.h>
#include <LibLsp/JsonRpc/TcpServer.h>
#include <LibLsp/JsonRpc/Condition.h>

// header for requests
#include <LibLsp/lsp/textDocument/hover.h>
#include <LibLsp/lsp/general/initialize.h>
#include <LibLsp/lsp/general/shutdown.h>
#include <LibLsp/lsp/textDocument/declaration_definition.h>

//header for notifs
#include <LibLsp/lsp/general/exit.h>
#include <LibLsp/lsp/general/initialized.h>
#include <LibLsp/lsp/textDocument/did_open.h>
#include <LibLsp/lsp/textDocument/did_change.h>
#include <LibLsp/lsp/textDocument/did_save.h>

//everything else
#include <functional>
#include <cctype>
#include <unordered_map>

namespace AsymptoteLsp
{
  template<typename TLeft, typename TRight>
  inline std::optional<std::pair<std::optional<TLeft>, std::optional<TRight>>> opt_left(TLeft const& opt)
  {
    return std::make_optional(std::make_pair(opt, std::nullopt));
  }

  template<typename TLeft, typename TRight>
  inline std::optional<std::pair<std::optional<TLeft>, std::optional<TRight>>> opt_right(TRight const& opt)
  {
    return std::make_optional(std::make_pair(std::nullopt, opt));
  }

  TextDocumentHover::Either fromString(std::string const &str);
  TextDocumentHover::Either fromMarkedStr(lsMarkedString const& markedString);
  TextDocumentHover::Either fromMarkedStr(std::string const& str, std::string const& language="asymptote");
  TextDocumentHover::Either fromMarkedStr(std::vector<std::string> const& stringList,
                                          std::string const& language="asymptote");

  string wslDos2Unix(std::string const& dosPath);
  string wslUnix2Dos(std::string const& unixPath);

  typedef std::unordered_map<std::string, std::unique_ptr<SymbolContext>> SymContextFilemap;

  class LspLog: public lsp::Log
  {
  public:
    void log(Level level, std::string&& msg) override;
    void log(Level level, const std::string& msg) override;
    void log(Level level, std::wstring&& msg) override;
    void log(Level level, const std::wstring& msg) override;
  };


  class AsymptoteLspServer: public lsp::TcpServer
  {
  public:
    AsymptoteLspServer(std::string const& addr, std::string const& port,
                       shared_ptr<lsp::ProtocolJsonHandler> const& jsonHandler,
                       shared_ptr<GenericEndpoint> const& endpoint, LspLog& log);
    ~AsymptoteLspServer();

    // copy constructors + copy assignment op
    AsymptoteLspServer(AsymptoteLspServer& sv) = delete;
    AsymptoteLspServer& operator=(AsymptoteLspServer const& sv) = delete;

    // move constructors and move assignment op
    AsymptoteLspServer(AsymptoteLspServer&& sv) = delete;
    AsymptoteLspServer& operator=(AsymptoteLspServer&& sv) = delete;

    void start();

  protected:
    td_hover::response handleHoverRequest(td_hover::request const&);
    td_initialize::response handleInitailizeRequest(td_initialize::request const&);
    td_shutdown::response handleShutdownRequest(td_shutdown::request const&);
    td_definition::response handleDefnRequest(td_definition::request const&);

    void onInitialized(Notify_InitializedNotification::notify& notify);
    void onExit(Notify_Exit::notify& notify);
    void onChange(Notify_TextDocumentDidChange::notify& notify);
    void onOpen(Notify_TextDocumentDidOpen::notify& notify);
    void onSave(Notify_TextDocumentDidSave::notify& notifY);

    void generateMissingTrees(std::string const& inputFile);

    void initializeRequestFn();
    void initializeNotifyFn();

    void reloadFile(std::string const&);
    SymbolContext* reloadFileRaw(std::string const&, bool const& fillTree=true);

    std::string plainFile;

  private:
    shared_ptr<lsp::ProtocolJsonHandler> pjh;
    shared_ptr<GenericEndpoint> ep;

    SymbolContext* plainCtx;

    LspLog& _log;

    unique_ptr<SymContextFilemap> symmapContextsPtr;
  };
}




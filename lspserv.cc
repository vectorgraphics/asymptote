//
// Created by Supakorn on 5/13/2021.
//


#include "common.h"
#include "lspserv.h"

#include <LibLsp/JsonRpc/stream.h>

#include <string>
#include <cstdlib>
#include <vector>
#include <memory>

#include <thread>

#include "dec.h"
#include "process.h"

#define GC_THREADS
#include "gc.h"


#define REGISTER_REQ_FN(typ, fn) remote_end_point_.registerRequestHandler(\
  [this](typ::request const& req) { return this->fn(req); });
#define REGISTER_NOTIF_FN(typ, handler) remote_end_point_.registerNotifyHandler(\
  [this](typ::notify& notif) { this->handler(notif); });



namespace AsymptoteLsp
{
  using std::unique_ptr;
  using std::shared_ptr;

  using absyntax::block;

  string wslDos2Unix(std::string const& dosPath)
  {
    bool isDrivePath = false;
    char drive;
    if (dosPath.length() >= 3)
    {
      if (dosPath[0] == '/' and dosPath[2] == ':') {
      isDrivePath = true;
      drive = dosPath[1];
      }
    }

    if (isDrivePath) {
      stringstream sstream;
      sstream << "/mnt/" << (char)tolower(drive) << dosPath.substr(3);
    return sstream.str();
    } else {
      return string(dosPath);
    }
  }

  TextDocumentHover::Either fromString(std::string const &str) {
    auto strobj=std::make_pair(std::make_optional(str), (std::optional<lsMarkedString>) std::nullopt);
    std::vector<decltype(strobj)> vec{strobj};
    return std::make_pair(vec, std::nullopt);
  }

  void LspLog::log(Level level, std::string &&msg) {
    cerr << msg << std::endl;
  }

  void LspLog::log(Level level, std::wstring &&msg) {
    std::wcerr << msg << std::endl;
  }

  void LspLog::log(Level level, const std::string &msg) {
    cerr << msg << std::endl;
  }

  void LspLog::log(Level level, const std::wstring &msg) {
    std::wcerr << msg << std::endl;
  }

  AsymptoteLspServer::AsymptoteLspServer(std::string const &addr, std::string const &port,
                                         shared_ptr<lsp::ProtocolJsonHandler> const &jsonHandler,
                                         shared_ptr<GenericEndpoint> const &endpoint, LspLog& log) :
          TcpServer(addr, port, jsonHandler, endpoint, log),
          pjh(jsonHandler), ep(endpoint), _log(log) {

    initializeRequestFn();
    initializeNotifyFn();

    symmap.clear();
  }

  void AsymptoteLspServer::initializeRequestFn()
  {
    REGISTER_REQ_FN(td_initialize, handleInitailizeRequest)
    REGISTER_REQ_FN(td_hover, handleHoverRequest)
    REGISTER_REQ_FN(td_shutdown, handleShutdownRequest)
  }

  void AsymptoteLspServer::initializeNotifyFn()
  {
    REGISTER_NOTIF_FN(Notify_InitializedNotification, onInitialized);
    REGISTER_NOTIF_FN(Notify_Exit, onExit);
    REGISTER_NOTIF_FN(Notify_TextDocumentDidChange, onChange);
    REGISTER_NOTIF_FN(Notify_TextDocumentDidOpen, onOpen);
    REGISTER_NOTIF_FN(Notify_TextDocumentDidSave, onSave);
  }

#pragma region notifications

  void AsymptoteLspServer::onInitialized(Notify_InitializedNotification::notify& notify)
  {
    cerr << "initialized" << endl;
  }

  void AsymptoteLspServer::onExit(Notify_Exit::notify& notify)
  {
    cerr << "exited" << endl;
    exit(0);
  }

  void AsymptoteLspServer::onChange(Notify_TextDocumentDidChange::notify& notify)
  {
    GC_stack_base gsb;
    GC_get_stack_base(&gsb);
    if (not GC_thread_is_registered())
    {
      GC_register_my_thread(&gsb);
    }

    cerr << "text change" << endl;
    lsDocumentUri fileUri(notify.params.textDocument.uri);
    string rawPath = settings::getSetting<bool>("wsl") ?
      wslDos2Unix(fileUri.GetRawPath()) : string(fileUri.GetRawPath());

    block* blk = ifile(rawPath.c_str()).getTree();
    if (blk != nullptr) {
      cerr << rawPath << endl;
      blk->createSymMap(symmap);
      cerr << symmap << endl;
    }

    GC_collect_a_little();
    GC_unregister_my_thread();
  }

  void AsymptoteLspServer::onOpen(Notify_TextDocumentDidOpen::notify& notify)
  {
    cerr << "text open" << endl;
  }

  void AsymptoteLspServer::onSave(Notify_TextDocumentDidSave::notify& notify)
  {
    cerr << "did save" << endl;
  }

#pragma endregion

  td_initialize::response AsymptoteLspServer::handleInitailizeRequest(td_initialize::request const &req)
  {
    td_initialize::response rsp;
    rsp.id = req.id;
    rsp.result.capabilities.hoverProvider=true;

    lsTextDocumentSyncOptions tdso;
    tdso.openClose = true;
    tdso.change = lsTextDocumentSyncKind::Full;
    lsSaveOptions so;
    so.includeText = true;
    tdso.save = so;
    rsp.result.capabilities.textDocumentSync = opt_right<lsTextDocumentSyncKind>(tdso);
    return rsp;
  }

  td_hover::response AsymptoteLspServer::handleHoverRequest(td_hover::request const& req)
  {
    cerr << "request for hover" << std::endl;
    td_hover::response rsp;
    rsp.result.contents=fromString("hello test signature.");
    return rsp;
  }

  td_shutdown::response AsymptoteLspServer::handleShutdownRequest(td_shutdown::request const &req)
  {
    std::cerr << "shut down" << std::endl;
    td_shutdown::response rsp;
    lsp::Any nullResp;
    JsonNull jn;
    nullResp.Set(jn);
    rsp.result=nullResp;
    this->stop();
    return rsp;
  }

  void AsymptoteLspServer::start() {
    this->run();
  }

  AsymptoteLspServer::~AsymptoteLspServer() {
    this->stop();
  }
}
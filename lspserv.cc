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
#include <queue>

#include <thread>

#include "dec.h"
#include "process.h"
#include "locate.h"

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

  string wslUnix2Dos(std::string const& unixPath)
  {
    bool isMntPath = false;
    char drive;

    char actPath[PATH_MAX];
    realpath(unixPath.c_str(), actPath);
    std::string fullPath(actPath);

    if (fullPath.length() >= 7) // /mnt/
    {
      if (fullPath.find("/mnt/") == 0)
      {
        isMntPath = true;
        drive = fullPath[5];
      }
    }

    if (isMntPath)
    {
      stringstream sstream;
      sstream << "/" << (char)tolower(drive) << ":" << fullPath.substr(6);
      return sstream.str();
    }
    else
    {
      return string(fullPath);
    }
  }

  TextDocumentHover::Either fromString(std::string const& str)
  {
    auto strobj=std::make_pair(std::make_optional(str), (std::optional<lsMarkedString>) std::nullopt);
    std::vector<decltype(strobj)> vec{strobj};
    return std::make_pair(vec, std::nullopt);
  }

  TextDocumentHover::Either fromMarkedStr(lsMarkedString const& markedString)
  {
    auto strobj=std::make_pair((std::optional<std::string>)std::nullopt, std::make_optional(markedString));
    std::vector<decltype(strobj)> vec{strobj};
    return std::make_pair(vec, std::nullopt);
  }

  TextDocumentHover::Either fromMarkedStr(std::string const& str, std::string const& language)
  {
    lsMarkedString lms;
    lms.language = language;
    lms.value = str;
    return fromMarkedStr(lms);
  }

  void AsymptoteLspServer::generateMissingTrees(std::string const& inputFile)
  {
    using extRefMap = std::unordered_map<std::string, SymbolContext*>;
    //using extRefMapLoc = std::pair<extRefMap*, std::string>;
    std::queue<extRefMap::iterator> procList;
    std::unordered_set<std::string> processing;

    processing.emplace(inputFile);
    SymbolContext* ctx = symmapContextsPtr->at(inputFile).get();

    for (auto const& locPair : ctx->getEmptyRefs())
    {
      procList.emplace(locPair);
    }

    // standard BFS algorithm
    while (not procList.empty())
    {
      auto it = procList.front();
      procList.pop();

      std::string filename(it->first);
      processing.emplace(filename);

      auto mapIt = symmapContextsPtr->find(filename);
      if (mapIt != symmapContextsPtr->end())
      {
        it->second = mapIt->second.get();
      }
      else
      {
        auto [fit, success] = symmapContextsPtr->emplace(filename,
                                       std::make_unique<SymbolContext>(posInFile(1, 1), filename));
        block* blk = ifile(filename.c_str()).getTree();
        blk->createSymMap(fit->second.get()); // parse symbol from there.

        // set plain.asy to plain
        if (plainCtx != nullptr)
        {
          fit->second->extFileRefs[plainFile] = plainCtx;
        }

        // also parse its neighbors
        for (auto const& sit : fit->second->getEmptyRefs())
        {
          if (processing.find(sit->first) == processing.end())
          {
            procList.emplace(sit);
          }
          else
          {
            // import cycles detected!
            cerr << "import cycles detected!" << endl;
          }
        }
        it->second = fit->second.get();
      }
    }
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
  }

  void AsymptoteLspServer::initializeRequestFn()
  {
    REGISTER_REQ_FN(td_initialize, handleInitailizeRequest)
    REGISTER_REQ_FN(td_hover, handleHoverRequest)
    REGISTER_REQ_FN(td_shutdown, handleShutdownRequest)
    REGISTER_REQ_FN(td_definition, handleDefnRequest)
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

    // string plain(settings::locateFile("plain.asy", true));
    // reloadFileRaw(static_cast<std::string>(plain));
  }

  void AsymptoteLspServer::onExit(Notify_Exit::notify& notify)
  {
    cerr << "exited" << endl;
    exit(0);
  }

  void AsymptoteLspServer::reloadFile(std::string const& filename)
  {
    string rawPath = settings::getSetting<bool>("wsl") ? wslDos2Unix(filename) : string(filename);
    reloadFileRaw(static_cast<std::string>(rawPath));
  }

  SymbolContext* AsymptoteLspServer::reloadFileRaw(std::string const& rawPath, bool const& fillTree)
  {
    block* blk = ifile(rawPath.c_str()).getTree();
    if (blk != nullptr) {
      auto it = symmapContextsPtr->find(rawPath);
      if (it != symmapContextsPtr->end())
      {
        it->second->reset(rawPath);
      }
      else
      {
        auto [fit, success] = symmapContextsPtr->emplace(
                rawPath, std::make_unique<SymbolContext>(posInFile(1, 1), rawPath));
        it = fit;
      }

      SymbolContext* newPtr = it->second.get();

      cerr << rawPath << endl;
      blk->createSymMap(newPtr);

      if (plainCtx != nullptr)
      {
        it->second->extFileRefs[plainFile] = plainCtx;
      } else if (rawPath == plainFile)
      {
        it->second->extFileRefs[plainFile] = newPtr;
      }

      if (fillTree)
      {
        generateMissingTrees(rawPath);
      }
      return it->second.get();
    }
    else
    {
      return nullptr;
    }
  }

  void AsymptoteLspServer::onChange(Notify_TextDocumentDidChange::notify& notify)
  {
    cerr << "text change" << endl;
  }

  void AsymptoteLspServer::onOpen(Notify_TextDocumentDidOpen::notify& notify)
  {
    cerr << "text open" << endl;
    lsDocumentUri fileUri(notify.params.textDocument.uri);
    reloadFile(fileUri.GetRawPath());
  }

  void AsymptoteLspServer::onSave(Notify_TextDocumentDidSave::notify& notify)
  {
    cerr << "did save" << endl;
    lsDocumentUri fileUri(notify.params.textDocument.uri);
    reloadFile(fileUri.GetRawPath());
  }

#pragma endregion

  td_initialize::response AsymptoteLspServer::handleInitailizeRequest(td_initialize::request const& req)
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

    rsp.result.capabilities.definitionProvider = std::make_pair(true, std::nullopt);

    // when starting the thread, memory is copied but not done correctly (why?)
    // hence, symmapContextsPtr gets assigned junk memory and we have to "clear" it
    // before we can give it a value.
    // FIXME: Investigate why. And also fix this
    symmapContextsPtr.release();
    plainCtx=nullptr;

    symmapContextsPtr=std::make_unique<SymContextFilemap>();

    plainFile=settings::locateFile("plain", true);
    plainCtx=reloadFileRaw(plainFile, false);
    generateMissingTrees(plainFile);
    return rsp;
  }

  td_hover::response AsymptoteLspServer::handleHoverRequest(td_hover::request const& req)
  {
    td_hover::response rsp;
    lsDocumentUri fileUri(req.params.textDocument.uri);
    string rawPath = settings::getSetting<bool>("wsl") ?
                     wslDos2Unix(fileUri.GetRawPath()) : string(fileUri.GetRawPath());
    auto rawPathStr = static_cast<std::string>(rawPath);
    auto fileSymIt = symmapContextsPtr->find(rawPathStr);
    std::vector<std::pair<optional<std::string>, optional<lsMarkedString>>> nullVec;

    if (fileSymIt == symmapContextsPtr->end())
    {
      rsp.result.contents.first = nullVec;
      return rsp;
    }

    auto [st, ctx]=fileSymIt->second->searchSymbol(fromLsPosition(req.params.position));
    if (not st.has_value())
    {
      rsp.result.contents.first = nullVec;
      return rsp;
    }

    auto [symText, startPos, endPos] = st.value();
    rsp.result.range=std::make_optional(lsRange(toLsPosition(startPos), toLsPosition(endPos)));

    auto typ = ctx->searchVarSignature(symText);
    rsp.result.contents=fromMarkedStr(typ.value_or("<decl-unknown> " + symText + ";"));
    return rsp;
  }

  td_shutdown::response AsymptoteLspServer::handleShutdownRequest(td_shutdown::request const& req)
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

  td_definition::response AsymptoteLspServer::handleDefnRequest(td_definition::request const& req)
  {
    td_definition::response rsp;
    rsp.result.first = make_optional(std::vector<lsLocation>());

    lsDocumentUri fileUri(req.params.textDocument.uri);
    string rawPath = settings::getSetting<bool>("wsl") ?
                     wslDos2Unix(fileUri.GetRawPath()) : string(fileUri.GetRawPath());

    auto rawPathStr = static_cast<std::string>(rawPath);
    auto fileSymIt = symmapContextsPtr->find(rawPathStr);
    if (fileSymIt != symmapContextsPtr->end())
    {
      auto [st, ctx]=fileSymIt->second->searchSymbol(fromLsPosition(req.params.position));
      if (st.has_value())
      {
        std::string sym(std::get<0>(st.value()));
        std::optional<posRangeInFile> posRange = ctx->searchVarDeclFull(sym);
        if (posRange.has_value())
        {
          auto [fil, posBegin, posEnd] = posRange.value();
          lsRange rng(toLsPosition(posBegin), toLsPosition(posEnd));

          std::string filReturn(
                  settings::getSetting<bool>("wsl") ? static_cast<std::string>(wslUnix2Dos(fil)) : fil);

          lsDocumentUri uri(filReturn);
          lsLocation loc(uri, rng);
          rsp.result.first->push_back(loc);
        }
      }
    }
    return rsp;
  }

  void AsymptoteLspServer::start() {
    this->run();
  }

  AsymptoteLspServer::~AsymptoteLspServer() {
    this->stop();
  }
}
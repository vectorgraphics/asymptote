#pragma once

#include "common.h"

#include <LibLsp/lsp/lsPosition.h>

#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <utility>

namespace AsymptoteLsp
{
  struct SymbolContext;

  typedef std::pair<std::string, SymbolContext*> contextedSymbol;
  typedef std::pair<size_t, size_t> posInFile;
  typedef std::pair<std::string, posInFile> filePos;
  typedef std::tuple<std::string, posInFile, posInFile> posRangeInFile;

  // NOTE: lsPosition is zero-indexed, while all Asymptote positions (incl this struct) is 1-indexed.
  inline posInFile fromLsPosition(lsPosition const& inPos)
  {
    return std::make_pair(inPos.line + 1, inPos.character + 1);
  }

  inline lsPosition toLsPosition(posInFile const& inPos)
  {
    return lsPosition(inPos.first - 1, inPos.second - 1);
  }

  inline bool posLt(posInFile const& p1, posInFile const& p2)
  {
    return (p1.first < p2.first) or ((p1.first == p2.first) and (p1.second < p2.second));
  }

  using std::unordered_map;

  // filename to positions
  struct positions
  {
    std::unordered_map<std::string, std::vector<posInFile>> pos;

    positions() = default;
    explicit positions(filePos const& positionInFile);
    void add(filePos const& positionInFile);
  };

  struct SymbolMaps
  {
    // FIXME: Factor in context as well, for example,
    // int x = 3;
    // x = 4; // referes to x=3 line.
    // for (...) {
    //   int x = 5;
    //   x = 7; // refers to x=5 line.
    // }
    // LSP needs to be able to differentiate between these two symbols and their usage.
    // a possible solution is context, which is a tree and a each context as a pointer to that node.

    unordered_map <std::string, posInFile> varDec;
    unordered_map <std::string, posInFile> funDec;

    // can refer to other files
    unordered_map <std::string, positions> varUsage;

    // python equivalent of dict[str, list[tuple(pos, sym)]]
    // filename -> list[(position, symbol)]

    std::vector<std::pair<posInFile, std::string>> usageByLines;

    inline void clear()
    {
      varDec.clear();
      varUsage.clear();
      usageByLines.clear();
    }

    std::optional<posRangeInFile> searchSymbol(posInFile const& inputPos);

  private:
    friend ostream& operator<<(std::ostream& os, const SymbolMaps& sym);
  };

  struct SymbolContext
  {
    posInFile contextLoc;
    SymbolContext* parent;
    SymbolMaps symMap;

    std::vector<std::unique_ptr<SymbolContext>> subContexts;

    SymbolContext():
      parent(nullptr) {
      std::cerr << "created symbol context";
    }

    virtual ~SymbolContext() = default;

    explicit SymbolContext(posInFile loc):
      contextLoc(std::move(loc)), parent(nullptr) {}

    SymbolContext(posInFile loc, SymbolContext* contextParent):
      contextLoc(std::move(loc)), parent(contextParent) {}

    template<typename T=SymbolContext, typename=std::enable_if<std::is_base_of<SymbolContext, T>::value>>
    T* newContext(posInFile const& loc)
    {
      subContexts.push_back(std::make_unique<T>(loc, this));
      return static_cast<T*>(subContexts.at(subContexts.size() - 1).get());
    }

    // [file, start, end]
    std::pair<std::optional<posRangeInFile>, SymbolContext*> searchSymbol(posInFile const& inputPos);

    virtual std::optional<posRangeInFile> searchVarDecl(std::string const& symbol);
  };

  struct AddDeclContexts: SymbolContext
  {
    unordered_map <std::string, posInFile> additionalDecs;
    AddDeclContexts(): SymbolContext() {}

    explicit AddDeclContexts(posInFile loc):
      SymbolContext(loc) {}

    AddDeclContexts(posInFile loc, SymbolContext* contextParent):
      SymbolContext(loc, contextParent) {}

    ~AddDeclContexts() override = default;

    std::optional<posRangeInFile> searchVarDecl(std::string const& symbol) override;
  };
}
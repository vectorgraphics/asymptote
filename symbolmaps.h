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

  struct SymbolInfo
  {
    std::string name;
    std::optional<std::string> type;
    posInFile pos;
    // std::optional<size_t> array_dim;

    SymbolInfo(std::string inName, posInFile position):
      name(std::move(inName)), type(nullopt), pos(std::move(position)) {}

    SymbolInfo(std::string inName, std::string inType, posInFile position):
      name(std::move(inName)), type(std::move(inType)), pos(std::move(position)) {}

    virtual ~SymbolInfo() = default;

    bool operator==(SymbolInfo const& sym) const;

    virtual std::string signature() const;
  };

  struct FunctionInfo: SymbolInfo
  {
    std::string returnType;
    using typeName = std::pair<std::string, std::optional<std::string>>;
    std::vector<typeName> arguments;
    std::optional<typeName> restArgs;

    FunctionInfo(std::string name, posInFile pos, std::string returnTyp):
            SymbolInfo(std::move(name), std::move(pos)),
            returnType(std::move(returnTyp)), arguments(), restArgs(nullopt) {}

    ~FunctionInfo() override = default;

    std::string signature() const override;
  };

  struct SymbolMaps
  {
    unordered_map <std::string, SymbolInfo> varDec;
    unordered_map <std::string, FunctionInfo> funDec;

    // can refer to other files
    unordered_map <std::string, positions> varUsage;

    // python equivalent of dict[str, list[tuple(pos, sym)]]
    // filename -> list[(position, symbol)]

    std::vector<std::pair<posInFile, std::string>> usageByLines;

    inline void clear()
    {
      varDec.clear();
      funDec.clear();
      varUsage.clear();
      usageByLines.clear();
    }

    std::optional<posRangeInFile> searchSymbol(posInFile const& inputPos);

  private:
    friend ostream& operator<<(std::ostream& os, const SymbolMaps& sym);
  };

  struct SymbolContext
  {
    std::optional<std::string> fileLoc;
    posInFile contextLoc;
    SymbolContext* parent;
    SymbolMaps symMap;

    // file interactions
    // access -> (file, id)
    // unravel -> id
    // include -> file
    // import = acccess + unravel

    using extRefMap = std::unordered_map<std::string, SymbolContext*>;
    extRefMap extFileRefs;
    std::unordered_map<std::string, std::string> fileIdPair;
    std::unordered_set<std::string> includeVals;
    std::unordered_set<std::string> unravledVals;
    std::vector<std::unique_ptr<SymbolContext>> subContexts;

    SymbolContext():
      parent(nullptr) {
      std::cerr << "created symbol context";
    }

    virtual ~SymbolContext() = default;

    explicit SymbolContext(posInFile loc);
    explicit SymbolContext(posInFile loc, std::string filename);

    SymbolContext(posInFile loc, SymbolContext* contextParent):
      fileLoc(nullopt), contextLoc(std::move(loc)), parent(contextParent)
    {
    }

    template<typename T=SymbolContext, typename=std::enable_if<std::is_base_of<SymbolContext, T>::value>>
    T* newContext(posInFile const& loc)
    {
      subContexts.push_back(std::make_unique<T>(loc, this));
      return static_cast<T*>(subContexts.at(subContexts.size() - 1).get());
    }

    // [file, start, end]
    virtual std::pair<std::optional<posRangeInFile>, SymbolContext*> searchSymbol(posInFile const& inputPos);

    virtual std::optional<posRangeInFile> searchVarDeclExt(
            std::string const& symbol, std::unordered_set<SymbolContext*>& searched);

    std::optional<posRangeInFile> searchVarDeclFull(std::string const& symbol);


    virtual std::optional<posRangeInFile> searchVarDecl(std::string const& symbol);
    virtual std::optional<std::string> searchVarSignature(std::string const& symbol) const;

    virtual std::list<extRefMap::iterator> getEmptyRefs()
    {
      std::list<extRefMap::iterator> finalList;

      for (auto it = extFileRefs.begin(); it != extFileRefs.end(); it++)
      {
        if (it->second == nullptr)
        {
          // cerr << it->first << endl;
          finalList.emplace_back(it);
        }
      }

      for (auto& ctx : subContexts)
      {
        finalList.splice(finalList.end(), ctx->getEmptyRefs());
      }

      return finalList;
    }

    std::optional<std::string> getFileName() const
    {
      if (fileLoc.has_value())
      {
        return fileLoc;
      }
      else
      {
        return parent == nullptr ? fileLoc : parent->getFileName();
      }
    }

    SymbolContext* getParent()
    {
      return parent == nullptr ? this : parent->getParent();
    }

    bool addEmptyExtRef(std::string const& fileName)
    {
      auto [it, success] = extFileRefs.emplace(fileName, nullptr);
      return success;
    }

    void reset(std::string const& newFile)
    {
      fileLoc = newFile;
      contextLoc = std::make_pair(1,1);
      clear();
    }

    void clear()
    {
      parent = nullptr;
      symMap.clear();
      clearExtRefs();
      subContexts.clear();
    }

    void clearExtRefs()
    {
      extFileRefs.clear();
      fileIdPair.clear();
      includeVals.clear();
      unravledVals.clear();
    }

  protected:
    std::optional<posRangeInFile> _searchVarDeclFull(
            std::string const& symbol, std::unordered_set<SymbolContext*>& searched);

  };

  struct AddDeclContexts: SymbolContext
  {
    unordered_map <std::string, SymbolInfo> additionalDecs;
    AddDeclContexts(): SymbolContext() {}

    explicit AddDeclContexts(posInFile loc):
      SymbolContext(loc) {}

    AddDeclContexts(posInFile loc, SymbolContext* contextParent):
      SymbolContext(loc, contextParent) {}

    ~AddDeclContexts() override = default;

    std::optional<posRangeInFile> searchVarDecl(std::string const& symbol) override;
    std::optional<std::string> searchVarSignature(std::string const& symbol) const override;
  };
}
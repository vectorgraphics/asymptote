#include "common.h"
#include "symbolmaps.h"
#include "locate.h"
#include <unordered_map>

namespace AsymptoteLsp
{
  std::string getPlainFile()
  {
    return std::string(settings::locateFile("plain", true));
  }

  positions::positions(filePos const& positionInFile)
  {
    add(positionInFile);
  }

  void positions::add(const filePos &positionInFile)
  {
    auto fileLoc=pos.find(positionInFile.first);
    if (fileLoc == pos.end())
    {
      pos.emplace(positionInFile.first, std::vector{positionInFile.second});
    } else
    {
      fileLoc->second.push_back(positionInFile.second);
    }
  }

  ostream& operator <<(std::ostream& os, const SymbolMaps& sym)
  {
    os << "var decs:" << endl;
    for (auto const& [key, value] : sym.varDec)
    {
      os << key << " " << value.pos.first << ":" << value.pos.second << endl;
    }
    return os;
  }

  std::optional<posRangeInFile> SymbolMaps::searchSymbol(posInFile const& inputPos)
  {
    // FIXME: can be optimized by binary search.
    for (auto const& [pos, sy] : usageByLines)
    {
      size_t endCharacter = pos.second + sy.length() - 1;
      bool posMatches =
              pos.first == inputPos.first and
              pos.second <= inputPos.second and
              inputPos.second <= endCharacter;
      bool isOperator = sy.find("operator ") == 0;
      if (posMatches and !isOperator)
      {
        posInFile endPos(pos.first, endCharacter + 1);
        return std::make_optional(std::make_tuple(sy, pos, endPos));
      }
    }
    return std::nullopt;
  }

  FunctionInfo& SymbolMaps::addFunDef(
          std::string const& funcName, posInFile const& position, std::string const& returnType)
  {
    auto [fit, _] = funDec.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(funcName), std::forward_as_tuple());

    auto& vit = fit->second.emplace_back(funcName, position, returnType);
    return vit;
  }

  std::pair<std::optional<posRangeInFile>, SymbolContext*> SymbolContext::searchSymbol(posInFile const& inputPos)
  {
    auto currCtxSym = symMap.searchSymbol(inputPos);
    if (currCtxSym.has_value())
    {
      return make_pair(currCtxSym, this);
    }
    // else, not found in currCtx;
    for (auto& subContext : subContexts)
    {
      if (!posLt(inputPos, subContext->contextLoc))
      {
        auto [subCtxSym, ctx]=subContext->searchSymbol(inputPos);
        if (subCtxSym.has_value())
        {
          return make_pair(subCtxSym, ctx);
        }
      }
    }
    return make_pair(std::nullopt, nullptr);
  }

  std::optional<std::string> SymbolContext::searchVarSignature(std::string const& symbol) const
  {
    auto pt = symMap.varDec.find(symbol);
    if (pt != symMap.varDec.end())
    {
      return pt->second.signature();
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarSignature(symbol) : nullopt;
  }

  std::optional<posRangeInFile> SymbolContext::searchVarDecl(std::string const& symbol)
  {
    return searchVarDecl(symbol, nullopt);
  }

  std::optional<posRangeInFile> SymbolContext::searchVarDecl(
          std::string const& symbol, std::optional<posInFile> const& position)
  {
    auto pt = symMap.varDec.find(symbol);
    if (pt != symMap.varDec.end())
    {
      auto [line, ch] = pt->second.pos;
      if ((not position.has_value()) or (!posLt(position.value(), pt->second.pos)))
      {
        return std::make_tuple(getFileName().value_or(""),
                               pt->second.pos, std::make_pair(line, ch + symbol.length()));
      }
    }

    auto ptFn = symMap.funDec.find(symbol);
    if (ptFn != symMap.funDec.end() and !ptFn->second.empty())
    {
      // FIXME: Right now, we have no way of knowing the exact position of the
      //        start of where the function name is. As an example, we do not know
      //        where the exact position of
      //        real testFunction(...) { ...
      //             ^
      auto ptValue = ptFn->second[0];
      if ((not position.has_value()) or (!posLt(position.value(), ptValue.pos)))
      {
        return std::make_tuple(getFileName().value_or(""), ptValue.pos, ptValue.pos);
      }
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarDecl(symbol, position) : nullopt;
  }

  void SymbolContext::addPlainFile()
  {
    std::string plainFile = getPlainFile();
    addEmptyExtRef(plainFile);
    unravledVals.emplace(plainFile);
  }

  SymbolContext::SymbolContext(posInFile loc):
    fileLoc(nullopt), contextLoc(std::move(loc)), parent(nullptr)
  {
    addPlainFile();
  }

  SymbolContext::SymbolContext(posInFile loc, std::string filename):
          fileLoc(std::move(filename)), contextLoc(std::move(loc)), parent(nullptr)
  {
    addPlainFile();
  }

  std::optional<posRangeInFile>
  SymbolContext::searchVarDeclFull(std::string const& symbol, std::optional<posInFile> const& position)
  {
    std::unordered_set<SymbolContext*> searched;
    return _searchVarDeclFull(symbol, searched, position);
  }

  std::optional<posRangeInFile>
  SymbolContext::_searchVarDeclFull(std::string const& symbol,
                                    std::unordered_set<SymbolContext*>& searched,
                                    std::optional<posInFile> const& position)
  {
    auto [it, notSearched] = searched.emplace(this->getParent());
    if (not notSearched)
    {
      // a loop in the search path. Stop now.
      return nullopt;
    }

    // local search first
    auto returnVal=searchVarDecl(symbol, position);
    return returnVal.has_value() ? returnVal : searchVarDeclExt(symbol, searched);
  }

  std::optional<posRangeInFile>
  SymbolContext::searchVarDeclExt(std::string const& symbol, std::unordered_set<SymbolContext*>& searched)
  {
    std::unordered_set<std::string> traverseSet;
    traverseSet.insert(unravledVals.begin(), unravledVals.end());
    traverseSet.insert(includeVals.begin(), includeVals.end());

    for (auto const& traverseVal : traverseSet)
    {
      if (traverseVal == this->getFileName())
      {
        continue;
      }
      auto returnValF = extFileRefs.at(traverseVal)->_searchVarDeclFull(symbol, searched);
      if (returnValF.has_value())
      {
        return returnValF;
      }
    }
    return nullopt;
  }

  std::list<SymbolContext::extRefMap::iterator> SymbolContext::getEmptyRefs()
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

  std::optional<std::string> SymbolContext::getFileName() const
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

  std::optional<std::string>
  SymbolContext::searchVarSignatureFull(std::string const& symbol)
  {
    std::unordered_set<SymbolContext*> searched;
    return _searchVarSignatureFull(symbol, searched);
  }

  std::optional<std::string>
  SymbolContext::_searchVarSignatureFull(std::string const& symbol, std::unordered_set<SymbolContext*>& searched)
  {
    auto [it, notSearched] = searched.emplace(this->getParent());
    if (not notSearched)
    {
      // a loop in the search path. Stop now.
      return nullopt;
    }

    // local search first
    auto returnVal=searchVarSignature(symbol);
    return returnVal.has_value() ? returnVal : searchVarSignatureExt(symbol, searched);
  }

  std::optional<std::string>
  SymbolContext::searchVarSignatureExt(std::string const& symbol, std::unordered_set<SymbolContext*>& searched)
  {
    std::unordered_set<std::string> traverseSet(unravledVals);
    traverseSet.insert(includeVals.begin(), includeVals.end());

    for (auto const& traverseVal : traverseSet)
    {
      if (traverseVal == this->getFileName())
      {
        continue;
      }
      auto returnValF = extFileRefs.at(traverseVal)->_searchVarSignatureFull(symbol, searched);
      if (returnValF.has_value())
      {
        return returnValF;
      }
    }
    return nullopt;
  }

  std::list<std::string>
  SymbolContext::searchFuncSignatureExt(std::string const& symbol, std::unordered_set<SymbolContext*>& searched)
  {
    std::list<std::string> finalList;
    std::unordered_set<std::string> traverseSet(unravledVals);
    traverseSet.insert(includeVals.begin(), includeVals.end());

    for (auto const& traverseVal : traverseSet)
    {
      if (traverseVal == this->getFileName())
      {
        continue;
      }
      auto returnValF = extFileRefs.at(traverseVal)->_searchFuncSignatureFull(symbol, searched);
      finalList.splice(finalList.end(), returnValF);
    }
    return finalList;
  }



  std::list<std::string> SymbolContext::_searchFuncSignatureFull(std::string const& symbol,
                                                                 std::unordered_set<SymbolContext*>& searched)
  {
    auto [it, notSearched] = searched.emplace(this->getParent());
    if (not notSearched)
    {
      // a loop in the search path. Stop now.
      return std::list<std::string>();
    }

    // local search first
    auto returnVal=searchFuncSignature(symbol);
    returnVal.splice(returnVal.end(), searchFuncSignatureExt(symbol, searched));
    return returnVal;
  }

  std::list<std::string> SymbolContext::searchFuncSignature(std::string const& symbol)
  {
    std::list<std::string> funcSigs;
    auto pt = symMap.funDec.find(symbol);
    if (pt != symMap.funDec.end())
    {
      std::transform(pt->second.begin(), pt->second.end(),
                     std::back_inserter(funcSigs),
                     [](FunctionInfo const& fnInfo) {
        return fnInfo.signature();
      });
    }

    if (parent != nullptr)
    {
      funcSigs.splice(funcSigs.end(), parent->searchFuncSignature(symbol));
    }
    return funcSigs;
  }

  std::list<std::string> SymbolContext::searchFuncSignatureFull(std::string const& symbol)
  {
    std::unordered_set<SymbolContext*> searched;
    return _searchFuncSignatureFull(symbol, searched);
  }

  std::optional<posRangeInFile> AddDeclContexts::searchVarDecl(
          std::string const& symbol, std::optional<posInFile> const& position)
  {
    auto pt = additionalDecs.find(symbol);
    if (pt != additionalDecs.end())
    {
      auto [line, ch] = pt->second.pos;
      if ((not position.has_value()) or (!posLt(position.value(), pt->second.pos)))
      {
        return std::make_tuple(getFileName().value_or(""),
                               pt->second.pos,
                               std::make_pair(line, ch + symbol.length()));
      }
    }

    return SymbolContext::searchVarDecl(symbol, position);
  }

  std::optional<std::string> AddDeclContexts::searchVarSignature(std::string const& symbol) const
  {
    auto pt = additionalDecs.find(symbol);
    return pt == additionalDecs.end() ? SymbolContext::searchVarSignature(symbol) : pt->second.signature();
  }

  bool SymbolInfo::operator==(SymbolInfo const& sym) const
  {
    return name==sym.name and type==sym.type and pos == sym.pos;
  }

  std::string SymbolInfo::signature() const
  {
    return type.value_or("<decl-unknown>") + " " + name + ";";
  }

  std::string FunctionInfo::signature() const
  {
    std::stringstream ss;
    ss << returnType << " " << name << "(";
    for (auto it = arguments.begin(); it != arguments.end(); it++)
    {
      auto const& [argtype, argname] = *it;
      ss << argtype;
      if (argname.has_value())
      {
        ss << " " << argname.value();
      }

      if (std::next(it) != arguments.end())
      {
        ss << ",";
      }

      if (std::next(it) != arguments.end() or restArgs.has_value())
      {
        ss << " ";
      }
    }

    if (restArgs.has_value())
    {
      auto const& [argtype, argname] = restArgs.value();
      ss << "... " << argtype;

      if (argname.has_value())
      {
        ss << " " << argname.value();
      }
    }
    ss << ");";
    return ss.str();
  }
}
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

  optional<fullSymPosRangeInFile> SymbolMaps::searchSymbol(posInFile const& inputPos)
  {
    // FIXME: can be optimized by binary search.
    for (auto const& [pos, syLit] : usageByLines)
    {
      size_t endCharacter = pos.second + syLit.name.length() - 1;
      bool posMatches =
              pos.first == inputPos.first and
              pos.second <= inputPos.second and
              inputPos.second <= endCharacter;
      bool isOperator = syLit.name.find("operator ") == 0;
      if (posMatches and !isOperator)
      {
        posInFile endPos(pos.first, endCharacter + 1);
        return boost::make_optional(std::make_tuple(syLit, pos, endPos));
      }
    }
    return nullopt;
  }

  FunctionInfo& SymbolMaps::addFunDef(
          std::string const& funcName, posInFile const& position, std::string const& returnType)
  {
    auto [fit, _] = funDec.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(funcName), std::forward_as_tuple());

    auto& vit = fit->second.emplace_back(funcName, position, returnType);
    return vit;
  }

  std::pair<optional<fullSymPosRangeInFile>, SymbolContext*> SymbolContext::searchSymbol(posInFile const& inputPos)
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
    return make_pair(nullopt, nullptr);
  }

  optional<std::string> SymbolContext::searchVarSignature(std::string const& symbol) const
  {
    auto pt = symMap.varDec.find(symbol);
    if (pt != symMap.varDec.end())
    {
      return pt->second.signature();
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarSignature(symbol) : nullopt;
  }

  optional<std::string> SymbolContext::searchVarType(std::string const& symbol) const
  {
    auto pt = symMap.varDec.find(symbol);
    if (pt != symMap.varDec.end())
    {
      return pt->second.type;
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarType(symbol) : nullopt;
  }

  optional<posRangeInFile> SymbolContext::searchVarDecl(std::string const& symbol)
  {
    return searchVarDecl(symbol, nullopt);
  }

  optional<posRangeInFile> SymbolContext::searchVarDecl(
          std::string const& symbol, optional<posInFile> const& position)
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
    return _searchVarFull<posRangeInFile>(
            searched,
            [&symbol, &position](SymbolContext* ctx)
            {
              return ctx->searchVarDecl(symbol, position);
            });
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

  optional<std::string> SymbolContext::getFileName() const
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

  optional<std::string> SymbolContext::searchVarSignatureFull(std::string const& symbol)
  {
    std::unordered_set<SymbolContext*> searched;
    return _searchVarFull<std::string>(searched,
            [&symbol](SymbolContext const* ctx)
            {
              return ctx->searchVarSignature(symbol);
            });
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

  optional<SymbolContext*> SymbolContext::searchStructContext(std::string const& tyVal) const
  {
    auto stCtx = symMap.typeDecs.find(tyVal);
    if (stCtx != symMap.typeDecs.end())
    {
      if (auto* stDec = dynamic_cast<StructDecs*>(stCtx->second.get()))
      {
        return make_optional(stDec->ctx);
      }
    }

    return parent != nullptr ? parent->searchStructContext(tyVal) : nullopt;
  };

  optional<std::string> SymbolContext::searchLitSignature(SymbolLit const& symbol)
  {
    if (symbol.scopes.empty())
    {
      return searchVarSignatureFull(symbol.name);
    }
    else
    {
      SymbolContext* ctx = this;
      std::vector scopes(symbol.scopes);
      auto searchTyFn = [](SymbolContext* pctx, std::string const& sym)
              {
                std::unordered_set<SymbolContext*> searched;
                return pctx->_searchVarFull<std::string>(
                        searched,
                        [&sym](SymbolContext* ctx)
                        {
                          return ctx->searchVarType(sym);
                        });
              };

      std::optional<std::string> tyInfo=searchTyFn(this, scopes.back());
      scopes.pop_back();
      if (not tyInfo.has_value())
      {
        return nullopt;
      }

      std::function<std::optional<SymbolContext*>(SymbolContext*, std::string const&)> searchCtxFn;
      searchCtxFn = [&searchCtxFn](SymbolContext* pctx, std::string const& tyVal)
      {
        auto stCtx = pctx->symMap.typeDecs.find(tyVal);
        if (stCtx != pctx->symMap.typeDecs.end())
        {
          if (auto* stDec = dynamic_cast<StructDecs*>(stCtx->second.get()))
          {
            return std::make_optional(stDec->ctx);
          }
        }

        return pctx->parent != nullptr ? searchCtxFn(pctx->parent, tyVal) : nullopt;
      };

      ctx = searchCtxFn(ctx, tyInfo.value()).value_or(nullptr);
      for (auto it = scopes.rbegin(); it != scopes.rend(); it++)
      {
        if (ctx == nullptr)
        {
          return nullopt;
        }
        // FIXME: Impelemnt scope searching
        //        example:
        //        varx.vary.varz => go into varx's type context (struct or external file), repeat.
        //        struct and extfile handled very differently

        // get next variable declaration loc.
        // assumes struct, hence we do not search entire workspace
        auto locVarDec = ctx->symMap.varDec.find(*it);
        if (locVarDec == ctx->symMap.varDec.end() or not locVarDec->second.type.has_value())
        {
          // dead end :(
          return nullopt;
        }

        // get the struct context of the type found
        std::unordered_set<SymbolContext*> searched;
        ctx = ctx->_searchVarFull<SymbolContext*>(
                searched,
                [&searchCtxFn, tyName = locVarDec->second.type.value()](SymbolContext* pctx)
                {
                  return searchCtxFn(pctx, tyName);
                }).value_or(nullptr);

        // ctx = searchCtxFn(ctx, locVarDec->second.type.value()).value_or(nullptr);
      }
      return ctx != nullptr ? ctx->searchVarSignatureFull(symbol.name) : nullopt;
    }
  }

  optional<posRangeInFile> AddDeclContexts::searchVarDecl(
          std::string const& symbol, optional<posInFile> const& position)
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

  optional<std::string> AddDeclContexts::searchVarSignature(std::string const& symbol) const
  {
    auto pt = additionalDecs.find(symbol);
    return pt != additionalDecs.end() ? pt->second.signature() : SymbolContext::searchVarSignature(symbol);
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
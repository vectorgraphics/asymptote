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

  optional<posRangeInFile>
  SymbolContext::searchVarDeclFull(std::string const& symbol, optional<posInFile> const& position)
  {
    std::unordered_set<SymbolContext*> searched;
    return _searchVarFull<posRangeInFile>(
            searched,
            [&symbol, &position](SymbolContext* ctx)
            {
              return ctx->searchVarDecl(symbol, position);
            });
  }

  std::list<ExternalRefs::extRefMap::iterator> SymbolContext::getEmptyRefs()
  {
    std::list<ExternalRefs::extRefMap::iterator> finalList;

    for (auto it = extRefs.extFileRefs.begin(); it != extRefs.extFileRefs.end(); it++)
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
    for (auto const& traverseVal : createTraverseSet())
    {
      if (traverseVal == this->getFileName())
      {
        continue;
      }
      auto returnValF = extRefs.extFileRefs.at(traverseVal)->_searchFuncSignatureFull(symbol, searched);
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
      if (SymbolContext* ctx=searchLitContext(symbol))
      {
        // ctx is a struct declaration. should not search beyond this struct.
        auto varDec = ctx->symMap.varDec.find(symbol.name);
        return varDec != ctx->symMap.varDec.end() ?
               optional<std::string>(varDec->second.signature()) : nullopt;
      }
    }

    return nullopt;
  }

  std::list<std::string> SymbolContext::searchLitFuncSignature(SymbolLit const& symbol)
  {
    std::list<std::string> signatures;
    if (symbol.scopes.empty())
    {
      signatures.splice(signatures.end(), searchFuncSignatureFull(symbol.name));
    }
    else
    {
      if (SymbolContext* ctx=searchLitContext(symbol))
      {
        // ctx is a struct declaration. should not search beyond this struct.
        auto fnDecs = ctx->symMap.funDec.find(symbol.name);
        if (fnDecs != ctx->symMap.funDec.end())
        {
          std::transform(
                  fnDecs->second.begin(), fnDecs->second.end(),
                  std::back_inserter(signatures),
                  [&scopes=symbol.scopes](FunctionInfo const& fnInf)
                  {
                    return fnInf.signature(scopes);
                  });
        }
      }
    }

    return signatures;
  }

  optional<posRangeInFile> SymbolContext::searchLitPosition(
          SymbolLit const& symbol,
          optional<posInFile> const& position)
  {
    if (symbol.scopes.empty())
    {
      return searchVarDeclFull(symbol.name, position);
    }
    else
    {
      SymbolContext* ctx=searchLitContext(symbol);
      if (ctx != nullptr) // ctx is a struct declaration. should not search beyond this struct.
      {
        auto varDec = ctx->symMap.varDec.find(symbol.name);
        if (varDec != ctx->symMap.varDec.end())
        {
          auto [line, ch] = varDec->second.pos;
          if ((not position.has_value()) or (!posLt(position.value(), varDec->second.pos)))
          {
            return std::make_tuple(ctx->getFileName().value_or(""),
                                   varDec->second.pos, std::make_pair(line, ch + symbol.name.length()));
          }
        }

        auto fnDec = ctx->symMap.funDec.find(symbol.name);
        if (fnDec != ctx->symMap.funDec.end() and !fnDec->second.empty())
        {
          auto ptValue = fnDec->second[0];
          if ((not position.has_value()) or (!posLt(position.value(), ptValue.pos)))
          {
            return std::make_tuple(getFileName().value_or(""), ptValue.pos, ptValue.pos);
          }
        }
      }

      return nullopt;
    }
  }

  SymbolContext* SymbolContext::searchStructCtxFull(std::string const& symbol)
  {
    std::unordered_set<SymbolContext*> searched;
    optional<std::string> tyInfo=_searchVarFull<std::string>(
            searched,
            [&symbol](SymbolContext* ctx)
            {
              return ctx->searchVarType(symbol);
            });


    if (not tyInfo.has_value())
    {
      return nullptr;
    }

    searched.clear();
    return _searchVarFull<SymbolContext*>(
            searched,
            [&tyName=tyInfo.value()](SymbolContext* pctx)
            {
              return pctx->searchStructContext(tyName);
            }).value_or(nullptr);
  }

  optional<SymbolContext*> SymbolContext::searchAccessDecls(std::string const& accessVal)
  {
    // fileIdPair includes accessVals

    auto src=extRefs.fileIdPair.find(accessVal);
    if (src != extRefs.fileIdPair.end())
    {
      std::string& fileSrc=src->second;
      auto ctxMap=extRefs.extFileRefs.find(fileSrc);
      if (ctxMap != extRefs.extFileRefs.end() && ctxMap->second != nullptr)
      {
        return ctxMap->second;
      }
    }

    return parent != nullptr ? parent->searchAccessDecls(accessVal) : nullopt;
  }

  SymbolContext* SymbolContext::searchLitContext(SymbolLit const& symbol)
  {
    if (symbol.scopes.empty())
    {
      return this;
    }
    else
    {
      std::vector scopes(symbol.scopes);
      bool isStruct=false;

      // search in struct
      auto* ctx=searchStructCtxFull(scopes.back());
      if (ctx)
      {
        isStruct = true;
        scopes.pop_back();
      }
      else
      {
        // search in access declarations
        std::unordered_set<SymbolContext*> searched;
        ctx =_searchVarFull<SymbolContext*>(
                searched,
                [&lastAccessor=scopes.back()](SymbolContext* pctx)
                {
                  return pctx->searchAccessDecls(lastAccessor);
                }).value_or(nullptr);

        if (ctx)
        {
          scopes.pop_back();
        }
      }

      for (auto it = scopes.rbegin(); it != scopes.rend() and ctx != nullptr; it++)
      {
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
          return nullptr;
        }

        // get the struct context of the type found
        std::unordered_set<SymbolContext*> searched;
        ctx = ctx->_searchVarFull<SymbolContext*>(
                searched,
                [&tyName=locVarDec->second.type.value()](SymbolContext* pctx)
                {
                  return pctx->searchStructContext(tyName);
                }).value_or(nullptr);
      }
      return ctx;
    }
  }

  std::unordered_set<std::string> SymbolContext::createTraverseSet()
  {
    std::unordered_set<std::string> traverseSet(extRefs.includeVals);
    traverseSet.emplace(getPlainFile());
    for (auto const& unravelVal : extRefs.unraveledVals)
    {
      traverseSet.emplace(extRefs.fileIdPair.at(unravelVal));
    }
    return traverseSet;
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
    return signature(std::vector<std::string>());
  }

  std::string FunctionInfo::signature(std::vector<std::string> const& scopes) const
  {
    std::stringstream ss;
    ss << returnType << " ";
    for (auto it=scopes.crbegin(); it!=scopes.crend(); it++)
    {
      ss << *it << ".";
    }
    ss << name << "(";
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
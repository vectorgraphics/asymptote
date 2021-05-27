#include "common.h"
#include "symbolmaps.h"
#include <unordered_map>

namespace AsymptoteLsp
{
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

    auto ptFn = symMap.funDec.find(symbol);
    if (ptFn != symMap.funDec.end())
    {
      return ptFn->second.signature();
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarSignature(symbol) : nullopt;
  }

  std::optional<posRangeInFile> SymbolContext::searchVarDecl(std::string const& symbol)
  {
    auto pt = symMap.varDec.find(symbol);
    if (pt != symMap.varDec.end())
    {
      auto [line, ch] = pt->second.pos;
      return std::make_tuple(pt->first, pt->second.pos, std::make_pair(line, ch + symbol.length()));
    }

    auto ptFn = symMap.funDec.find(symbol);
    if (ptFn != symMap.funDec.end())
    {
      // FIXME: Right now, we have no way of knowing the exact position of the
      //        start of where the function name is. As an example, we do not know
      //        where the exact position of
      //        real testFunction(...) { ...
      //             ^
      return std::make_tuple(ptFn->first, ptFn->second.pos, ptFn->second.pos);
    }

    // otherwise, search parent.
    return parent != nullptr ? parent->searchVarDecl(symbol) : nullopt;
  }

  std::optional<posRangeInFile> AddDeclContexts::searchVarDecl(std::string const& symbol)
  {
    auto pt = additionalDecs.find(symbol);
    if (pt != additionalDecs.end())
    {
      auto [line, ch] = pt->second.pos;
      return std::make_tuple(pt->first, pt->second.pos, std::make_pair(line, ch + symbol.length()));
    }

    return SymbolContext::searchVarDecl(symbol);
  }

  std::optional<std::string> AddDeclContexts::searchVarSignature(std::string const& symbol) const
  {
    auto pt = additionalDecs.find(symbol);
    if (pt != additionalDecs.end())
    {
      return pt->second.signature();
    }

    return SymbolContext::searchVarSignature(symbol);
  }
}
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
      os << key << " " << value.first << ":" << value.second << endl;
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

  std::optional<posRangeInFile> SymbolContext::searchSymbol(posInFile const& inputPos)
  {
    auto currCtxSym = symMap.searchSymbol(inputPos);
    if (currCtxSym.has_value())
    {
      return currCtxSym;
    }

    // else, not found in currCtx;

    for (auto& subContext : subContexts)
    {
      if (!posLt(inputPos, subContext->contextLoc))
      {
        auto currCtxSym=subContext->searchSymbol(inputPos);
        if (currCtxSym.has_value())
        {
          return currCtxSym;
        }
      }
    }
    return std::nullopt;
  }
}
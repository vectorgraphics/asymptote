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
      os << key << " " << value.first << ":" << value.second.first << ":" << value.second.second << endl;
    }
    return os;
  }

  lineUsage::lineUsage(posInFile const& pos, std::string const& sym)
  {
    add(pos, sym);
  }
  // start, end position (non-inclusive)

  void lineUsage::add(posInFile const& pos, std::string const& sym)
  {
    usageByLine.emplace_back(pos, sym);
  }

  std::optional<std::tuple<std::string, posInFile, posInFile>> lineUsage::searchSymbol(posInFile const& inputPos)
  {
    // FIXME: can be optimized by binary search.
    for (auto const& [pos, sy] : usageByLine)
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
}
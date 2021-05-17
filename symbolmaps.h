#pragma once

#include "common.h"
#include <unordered_map>
#include <unordered_set>

namespace AsymptoteLsp
{
  typedef std::pair<size_t, size_t> posInFile;
  typedef std::pair<std::string, posInFile> filePos;

  // filename to positions
  typedef std::unordered_map<std::string, std::vector<posInFile>> positions;

  struct SymbolMaps
  {
    std::unordered_map <std::string, filePos> varDec;
    std::unordered_map <std::string, positions> varUsage;

    inline void clear()
    {
      varDec.clear();
      varUsage.clear();
    }

  private:
    friend ostream& operator<<(std::ostream& os, const SymbolMaps& sym)
    {
      os << "var decs:" << endl;
      for (auto const& [key, value] : sym.varDec)
      {
        os << key << " " << value.first << ":" << value.second.first << ":" << value.second.second << endl;
      }
      return os;
    }
  };
}
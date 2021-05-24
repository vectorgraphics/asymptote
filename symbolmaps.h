#pragma once

#include "common.h"

#include <LibLsp/lsp/lsPosition.h>

#include <unordered_map>
#include <unordered_set>
#include <tuple>

namespace AsymptoteLsp
{
  typedef std::pair<size_t, size_t> posInFile;
  typedef std::pair<std::string, posInFile> filePos;

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

  struct lineUsage
  {
    std::vector<std::pair<posInFile, std::string>> usageByLine;

    lineUsage() = default;
    lineUsage(posInFile const& pos, std::string const& sym);
    void add(posInFile const& pos, std::string const& sym);
    std::optional<std::tuple<std::string, posInFile, posInFile>> searchSymbol(posInFile const& inputPos);
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

    unordered_map <std::string, filePos> varDec;
    unordered_map <std::string, positions> varUsage;

    // python equivalent of dict[str, list[tuple(pos, sym)]]
    // filename -> list[(position, symbol)]
    unordered_map<std::string, lineUsage> usageByFiles;

    inline void clear()
    {
      varDec.clear();
      varUsage.clear();
      usageByFiles.clear();
    }

  private:
    friend ostream& operator<<(std::ostream& os, const SymbolMaps& sym);
  };
}
//
// Created by jamie on 8/23/21.
//

#pragma once

#include "common.h"

namespace camp
{

// Ensure the IBL image library is available: if locateFile("ibl") is empty,
// prompt (if possible) to download the current image into ~/.asy/ibl/.
// Returns the resolved ibl directory path (e.g. "/home/user/.asy/ibl/"),
// or "" if the library is unavailable and the user declined / no download
// possible.
string ensureIBLDir();

class IEXRFile
{
public:
  IEXRFile(string const& file);
  ~IEXRFile();

  [[nodiscard]]
  float const* getData() const {
    return data;
  }

  [[nodiscard]]
  std::pair<int,int> size() const {
    return std::make_pair(width, height);
  }

private:
  float* data=nullptr;
  int width=0, height=0;
};

} // namespace camp;

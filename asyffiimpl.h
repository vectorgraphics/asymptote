#pragma once

#include "asyffi.h"
#include "common.h"

namespace camp
{
class AsyArgsImpl : public IAsyArgs
{
public:
  [[nodiscard]]
  size_t getArgumentCount() const override;
  [[nodiscard]]
  double getNumberedArgAsReal(const size_t& argNum) const override;
  [[nodiscard]]
  int64_t getNumberedArgAsInt(const size_t& argNum) const override;
  
  void addArgs(void* arg)
  {
    argsStorage.push_back(arg);
  }
  
private:
  mem::vector<void*> argsStorage;
};

}
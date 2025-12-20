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
  IAsyItem* getNumberedArg(const size_t& argNum) const override;

  
  void addArgs(IAsyItem* arg);

private:
  mem::vector<IAsyItem*> argsStorage;
};

class AsyContextImpl: public IAsyContext
{
public:
  void* malloc(size_t const& size) override;
  void* mallocAtomic(size_t const& size) override;
};

}
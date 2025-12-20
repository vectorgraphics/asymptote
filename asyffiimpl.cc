#include "asyffiimpl.h"

#include "common.h"
#include "util.h"

namespace camp
{

size_t AsyArgsImpl::getArgumentCount() const { return argsStorage.size(); }

double AsyArgsImpl::getNumberedArgAsReal(const size_t& argNum) const
{
  return *static_cast<double*>(argsStorage.at(argNum));
}


void AsyArgsImpl::addArgs(IAsyItem* arg) { argsStorage.push_back(arg); }
void* AsyContextImpl::malloc(size_t const& size) { return asy_malloc(size); }
void* AsyContextImpl::mallocAtomic(size_t const& size)
{
  return asy_malloc_atomic(size);
}
}// namespace camp
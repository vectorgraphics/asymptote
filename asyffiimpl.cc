#include "asyffiimpl.h"

#include "util.h"

namespace camp
{

size_t AsyArgsImpl::getArgumentCount() const { return argsStorage.size(); }

double AsyArgsImpl::getNumberedArgAsReal(const size_t& argNum) const
{
  return *static_cast<double*>(argsStorage.at(argNum));
}

int64_t AsyArgsImpl::getNumberedArgAsInt(const size_t& argNum) const
{
  return *static_cast<int64_t*>(argsStorage.at(argNum));
}
}// namespace camp
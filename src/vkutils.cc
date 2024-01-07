#include "vkutils.h"

#if defined(HAVE_VULKAN)

#include "common.h"
#include "errormsg.h"

namespace camp
{
namespace vkutils
{

void checkVkResult(vk::Result const& result)
{
  if (result != vk::Result::eSuccess)
  {
    ostringstream oss;
    oss << "Vulkan operation failed; message: " << to_string(result);
    camp::reportError(oss);
  }
}

}
}
#endif
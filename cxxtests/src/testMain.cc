#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "common.h"

int main(int argc, char* argv[])
{
#if defined(USEGC)
  GC_init();
#endif

  ::testing::InitGoogleTest(&argc,argv);
  ::testing::InitGoogleMock(&argc,argv);
  return RUN_ALL_TESTS();
}

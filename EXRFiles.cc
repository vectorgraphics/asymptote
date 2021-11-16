//
// Created by jamie on 8/23/21.
//

#include "EXRFiles.h"

namespace camp
{
using std::cout;
using std::cerr;
using std::endl;

IEXRFile::IEXRFile(const string& file)
{
  char const* err;
  int ret;
  if((ret=LoadEXR(&data,&width,&height, file.c_str(),&err)) != TINYEXR_SUCCESS)
    {
    cerr << "TinyEXR Error: " << err << endl;
    FreeEXRErrorMessage(err);
    exit(-1);
  }
}

IEXRFile::~IEXRFile()
{
  if(data) {
    free(data);
    data=nullptr;
  }
}

}

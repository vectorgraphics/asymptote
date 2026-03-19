#include <sys/types.h>
#include <cstring>
#include <sstream>

#include "fftw++.h"

using namespace std;
using namespace utils;

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);

// User settings:
size_t fftw::effort=FFTW_MEASURE;
string wisdomName="wisdom3.txt";
ostringstream wisdomTemp;
size_t fftw::maxthreads=1;

fftw_plan (*fftw::planner)(fftw *f, Complex *in, Complex *out)=Planner;

const char *fftw::oddshift="Shift is not implemented for odd nx";
const char *inout=
  "constructor and call must be both in place or both out of place";

Mfft1d::Table Mfft1d::threadtable;
Mrcfft1d::Table Mrcfft1d::threadtable;
Mcrfft1d::Table Mcrfft1d::threadtable;

static size_t lastWisdomLength=0;

void loadWisdom()
{
  if(lastWisdomLength == 0) {
    wisdomTemp << wisdomName << "_" << getpid();
    ifstream ifWisdom;
    ifWisdom.open(wisdomName);
    ostringstream wisdom;
    wisdom << ifWisdom.rdbuf();
    ifWisdom.close();
    const string& s=wisdom.str();
    fftw_import_wisdom_from_string(s.c_str());
    lastWisdomLength=s.size();
  }
}

void saveWisdom()
{
  char *wisdom=fftw_export_wisdom_to_string();
  size_t len=strlen(wisdom);
  if(len > lastWisdomLength) {
    ofstream ofWisdom;
    ofWisdom.open(wisdomTemp.str().c_str());
    ofWisdom << wisdom;
    fftw_free(wisdom);
    ofWisdom.close();
    renameOverwrite(wisdomTemp.str().c_str(),wisdomName.c_str());
    lastWisdomLength=len;
  }
}

fftw_plan Planner(fftw *F, Complex *in, Complex *out)
{
  loadWisdom();
  fftw::effort |= FFTW_WISDOM_ONLY;
  fftw_plan plan=F->Plan(in,out);
  fftw::effort &= ~FFTW_WISDOM_ONLY;
  if(!plan) {
    plan=F->Plan(in,out);
    saveWisdom();
  }
  return plan;
}

ThreadBase::ThreadBase() {threads=fftw::maxthreads;}

}

namespace utils {
size_t defaultmpithreads=1;
}

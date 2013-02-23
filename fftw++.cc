#include "fftw++.h"

namespace fftwpp {

bool fftw::Wise=false;
bool fftw::autothreads=true;
const double fftw::twopi=2.0*acos(-1.0);

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *WisdomName="wisdom3.txt";
unsigned int fftw::maxthreads=1;
double fftw::testseconds=0.1; // Time limit for threading efficiency tests

void fftw::LoadWisdom() {
  std::ifstream ifWisdom;
  ifWisdom.open(WisdomName);
  fftwpp_import_wisdom(GetWisdom,ifWisdom);
  ifWisdom.close();
}

void fftw::SaveWisdom() {
  std::ofstream ofWisdom;
  ofWisdom.open(WisdomName);
  fftwpp_export_wisdom(PutWisdom,ofWisdom);
  ofWisdom.close();
}
  
}

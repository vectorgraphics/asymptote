#include "fftw++.h"

namespace fftwpp {

const double fftw::twopi=2.0*acos(-1.0);

fft1d::Table fft1d::threadtable;
mfft1d::Table mfft1d::threadtable;
rcfft1d::Table rcfft1d::threadtable;
crfft1d::Table crfft1d::threadtable;
mrcfft1d::Table mrcfft1d::threadtable;
mcrfft1d::Table mcrfft1d::threadtable;
fft2d::Table fft2d::threadtable;

// User settings:
unsigned int fftw::effort=FFTW_MEASURE;
const char *fftw::WisdomName=".wisdom";
unsigned int fftw::maxthreads=1;
double fftw::testseconds=1.0; // Time limit for threading efficiency tests
unsigned int fftw::Wise=0;
bool fftw::mpi=false;

const char *fftw::oddshift="Shift is not implemented for odd nx";
const char *inout="constructor and call must be both in place or both out of place";

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

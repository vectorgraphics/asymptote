/* Fast Fourier transform C++ header class for the FFTW3 Library
   Copyright (C) 2004-12 John C. Bowman, University of Alberta

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef __fftwpp_h__
#define __fftwpp_h__ 1

#define __FFTWPP_H_VERSION__ 1.14

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <fftw3.h>
#include <cerrno>

#ifndef FFTWPP_SINGLE_THREAD
#include <omp.h>
#endif

inline int get_thread_num() 
{
#ifdef FFTWPP_SINGLE_THREAD
  return 0;
#else
  return omp_get_thread_num();
#endif  
}

inline int get_max_threads() 
{
#ifdef FFTWPP_SINGLE_THREAD
  return 1;
#else
  return omp_get_max_threads();
#endif  
}

#ifndef __Complex_h__
#include <complex>
typedef std::complex<double> Complex;
#endif

#include "seconds.h"

#ifndef HAVE_POSIX_MEMALIGN

#ifdef __GLIBC_PREREQ
#if __GLIBC_PREREQ(2,3)
#define HAVE_POSIX_MEMALIGN
#endif
#else
#ifdef _POSIX_SOURCE
#define HAVE_POSIX_MEMALIGN
#endif
#endif

#endif

#ifdef __Array_h__

namespace Array {
static const array1<Complex> NULL1;  
static const array2<Complex> NULL2;  
static const array3<Complex> NULL3;
}

#else

#ifdef HAVE_POSIX_MEMALIGN
#ifdef _AIX
extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif
#else
namespace Array {

// Adapted from FFTW aligned malloc/free.  Assumes that malloc is at least
// sizeof(void*)-aligned. Allocated memory must be freed with free0.
inline int posix_memalign0(void **memptr, size_t alignment, size_t size)
{
  if(alignment % sizeof (void *) != 0 || (alignment & (alignment - 1)) != 0)
    return EINVAL;
  void *p0=malloc(size+alignment);
  if(!p0) return ENOMEM;
  void *p=(void *)(((size_t) p0+alignment)&~(alignment-1));
  *((void **) p-1)=p0;
  *memptr=p;
  return 0;
}

inline void free0(void *p)
{
  if(p) free(*((void **) p-1));
}

}
#endif

namespace Array {

template<class T>
inline void newAlign(T *&v, size_t len, size_t align)
{
  void *mem=NULL;
  const char *invalid="Invalid alignment requested";
  const char *nomem="Memory limits exceeded";
#ifdef HAVE_POSIX_MEMALIGN
  int rc=posix_memalign(&mem,align,len*sizeof(T));
#else  
  int rc=posix_memalign0(&mem,align,len*sizeof(T));
#endif  
  if(rc == EINVAL) std::cerr << invalid << std::endl;
  if(rc == ENOMEM) std::cerr << nomem << std::endl;
  v=(T *) mem;
  for(size_t i=0; i < len; i++) new(v+i) T;
}

template<class T>
inline void deleteAlign(T *v, size_t len)
{
  for(size_t i=len-1; i > 0; i--) v[i].~T();
  v[0].~T();
#ifdef HAVE_POSIX_MEMALIGN
  free(v);
#else
  free0(v);
#endif  
}
}

#endif

namespace fftwpp {

inline Complex *ComplexAlign(size_t size)
{
  Complex *v;
  Array::newAlign(v,size,sizeof(Complex));
  return v;
}

inline double *doubleAlign(size_t size)
{
  double *v;
  Array::newAlign(v,size,sizeof(Complex));
  return v;
}

template<class T>
inline void deleteAlign(T *p)
{
#ifdef HAVE_POSIX_MEMALIGN
  free(p);
#else
  Array::free0(p);
#endif  
}

// Obsolete names:
#define FFTWComplex ComplexAlign
#define FFTWdouble doubleAlign
#define FFTWdelete deleteAlign

inline void fftwpp_export_wisdom(void (*emitter)(char c, std::ofstream& s),
                                 std::ofstream& s)
{
  fftw_export_wisdom((void (*) (char, void *)) emitter,(void *) &s);
}

inline int fftwpp_import_wisdom(int (*g)(std::ifstream& s), std::ifstream &s)
{
  return fftw_import_wisdom((int (*) (void *)) g,(void *) &s);
}

inline void PutWisdom(char c, std::ofstream& s) {s.put(c);}
inline int GetWisdom(std::ifstream& s) {return s.get();}

// Base clase for fft routines
//
class fftw {
protected:
  unsigned int doubles; // number of double precision values in dataset
  int sign;
  double norm;

  fftw_plan plan;
  bool inplace;
  
  unsigned int Dist(unsigned int n, ptrdiff_t stride, ptrdiff_t dist) {
    return dist ? dist : ((stride == 1) ? n : 1);
  }
  
  unsigned int realsize(unsigned int n, Complex *in, Complex *out=NULL) {
    return (!out || in == out) ? 2*(n/2+1) : n;
  }
  
  unsigned int realsize(unsigned int n, Complex *in, double *out) {
    return realsize(n,in,(Complex *) out);
  }
  
  unsigned int realsize(unsigned int n, double *in, Complex *out) {
    return realsize(n,(Complex *) in,out);
  }
  
  static bool Wise;
  static const double twopi;
  unsigned int threads;
  
public:
  static unsigned int effort;
  static unsigned int maxthreads;
  static double testseconds;
  
  unsigned int Threads() {return threads;}
  
  // Shift the Fourier origin to (nx/2,0).
  static void Shift(Complex *data, unsigned int nx, unsigned int ny,
                    int sign=0) {
    const unsigned int nyp=ny/2+1;
    Complex *pstop=data+nx*nyp;
    if(nx % 2 == 0) {
      int pinc=2*nyp;
      for(Complex *p=data+nyp; p < pstop; p += pinc) {
        for(unsigned int j=0; j < nyp; j++) p[j]=-p[j];
      }
    } else {
      if(sign) {
        unsigned int c=nx/2;
        double arg=twopi*c/nx; 
        for(unsigned int i=0; i < nx; i++) {
          double iarg=i*arg;
          Complex zeta(cos(iarg),sign*sin(iarg));
          Complex *datai=data+i*nyp;
          for(unsigned int j=0; j < nyp; j++) datai[j] *= zeta;
        }
      } else {
        std::cerr << "Shift for odd nx must be signed and interleaved" 
                  << std::endl;
        exit(1);
      }
    }
  }

  // Shift the Fourier origin to (nx/2,ny/2,0).
  static void Shift(Complex *data, unsigned int nx, unsigned int ny,
                    unsigned int nz, int sign=0) {
    const unsigned int nzp=nz/2+1;
    const unsigned int nyzp=ny*nzp;
    if(nx % 2 == 0 && ny % 2 == 0) {
      const unsigned int pinc=2*nzp;
      Complex *pstop=data;
      Complex *p=data;
      for(unsigned i=0; i < nx; i++) {
        if(i % 2) p -= nzp;
        else p += nzp;
        pstop += nyzp;
        for(; p < pstop; p += pinc) {
          for(unsigned int k=0; k < nzp; k++) p[k]=-p[k];
        }
      }
    } else {
      if(sign) {
        unsigned int cx=nx/2;
        double argx=twopi*cx/nx;
        unsigned int cy=ny/2;
        double argy=twopi*cy/ny;
        for(unsigned i=0; i < nx; i++) {
          double iarg=i*argx;
          Complex zetax(cos(iarg),sign*sin(iarg));
          Complex *datai=data+nyzp*i;
          for(unsigned j=0; j < ny; j++) {
            double jarg=j*argy;
            Complex zeta=zetax*Complex(cos(jarg),sign*sin(jarg));
            Complex *dataij=datai+nzp*j;
            for(unsigned int k=0; k < nzp; k++) dataij[k] *= zeta;
          }
        }
      } else {
        std::cerr << "Shift for odd nx or ny must be signed and interleaved" 
                  << std::endl;
        exit(1);
      }
    }
  }

  static bool autothreads;
  fftw(unsigned int doubles, int sign, unsigned int n=0) : 
    doubles(doubles), sign(sign), norm(1.0/(n ? n : (doubles+1)/2)),
    plan(NULL) {
    if(autothreads) {
#ifndef FFTWPP_SINGLE_THREAD
      fftw_init_threads();
#endif      
      autothreads=false;
    }
  }
  
  virtual ~fftw() {if(plan) fftw_destroy_plan(plan);}
  
  virtual fftw_plan Plan(Complex *in, Complex *out)=0;
  
  inline void CheckAlign(Complex *p, const char *s) {
    if((size_t) p % sizeof(Complex) == 0) return;
    std::cerr << "WARNING: " << s << " array is not " << sizeof(Complex) 
              << "-byte aligned: address " << p << std::endl;
  }
  
  double stdev(unsigned int N, double sum, double sum2) {
    return N > 1 ? sqrt((sum2-sum*sum/N)/(N-1)) : 0.0;
  }
  
  void noplan() {
    std::cerr << "Unable to construct FFTW plan" << std::endl;
    exit(1);
  }
  
  void planThreads(unsigned int threads) {
#ifndef FFTWPP_SINGLE_THREAD
    omp_set_num_threads(threads);
    fftw_plan_with_nthreads(threads);
#endif    
  }
  
  void Setup(Complex *in, Complex *out=NULL) {
    if(!Wise) LoadWisdom();
    bool alloc=!in;
    if(alloc) in=ComplexAlign((doubles+1)/2);
#ifndef NO_CHECK_ALIGN    
    CheckAlign(in,"constructor input");
    if(out) CheckAlign(out,"constructor output");
    else out=in;
#else
    if(!out) out=in;
#endif    
    inplace=(out==in);
    
    planThreads(1);
    fftw_plan plan1=Plan(in,out);
    if(!plan1) noplan();
    plan=plan1;
    
    if(maxthreads > 1) {
      double sum2=0.0;
      unsigned int N=1;
      double begin=totalseconds();
      double lastseconds=begin;
      double stop=begin+testseconds;
      for(;;++N) {
        fft(in,out);
        double t=totalseconds();
        double seconds=t-lastseconds;
        sum2 += seconds*seconds;
        lastseconds=t;
        if(t > stop)
          break;
      }

      double end=totalseconds();
      double sum=end-begin;
      double mean1=sum/N;
      double stdev1=stdev(N,sum,sum2);
        
      threads=maxthreads;
      planThreads(threads);
      plan=Plan(in,out);
      if(!plan) noplan();

      if(plan) {
        double begin=totalseconds();
        double lastseconds=begin;
        double stop=begin+testseconds;
        N=1;
        for(;;++N) {
          fft(in,out);
          double t=totalseconds();
          double seconds=t-lastseconds;
          sum2 += seconds*seconds;
          lastseconds=t;
          if(t > stop)
            break;
        }
        double end=totalseconds();
        double sum=end-begin;
        double mean2=sum/N;
        
        if(mean2 > mean1-stdev1) {
          threads=1;
          fftw_destroy_plan(plan);
          plan=plan1;
        } else {
          fftw_destroy_plan(plan1);
          threads=maxthreads;
        }
      }
    } else
      threads=1;
    
    if(alloc) Array::deleteAlign(in,(doubles+1)/2);
    SaveWisdom();
  }
  
  void Setup(Complex *in, double *out) {Setup(in,(Complex *) out);}
  void Setup(double *in, Complex *out=NULL) {Setup((Complex *) in,out);}
  
  void LoadWisdom();
  void SaveWisdom();
  
  virtual void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft(plan,(fftw_complex *) in,(fftw_complex *) out);
  }
    
  Complex *Setout(Complex *in, Complex *out) {
#ifndef NO_CHECK_ALIGN    
    CheckAlign(in,"input");
    if(out) CheckAlign(out,"output");
    else out=in;
#else
    if(!out) out=in;
#endif    
    if(inplace ^ (out == in)) {
      std::cerr << "ERROR: fft constructor and call must be both in place or both out of place" << std::endl; 
      exit(1);
    }
    return out;
  }
  
  void fft(Complex *in, Complex *out=NULL) {
    out=Setout(in,out);
    Execute(in,out);
  }
    
  void fft(double *in, Complex *out=NULL) {
    fft((Complex *) in,out);
  }
  
  void fft(Complex *in, double *out) {
    fft(in,(Complex *) out);
  }
  
  void fft0(Complex *in, Complex *out=NULL) {
    out=Setout(in,out);
    Execute(in,out,true);
  }
    
  void fft0(double *in, Complex *out=NULL) {
    fft0((Complex *) in,out);
  }
  
  void fft0(Complex *in, double *out) {
    fft0(in,(Complex *) out);
  }
  
  void Normalize(Complex *out) {
    unsigned int stop=(doubles+1)/2;
    for(unsigned int i=0; i < stop; i++) out[i] *= norm;
  }

  void Normalize(double *out) {
    for(unsigned int i=0; i < doubles; i++) out[i] *= norm;
  }
  
  virtual void fftNormalized(Complex *in, Complex *out=NULL) {
    out=Setout(in,out);
    Execute(in,out);
    Normalize(out);
  }
  
  void fftNormalized(Complex *in, double *out) {
    out=(double *) Setout(in,(Complex *) out);
    Execute(in,(Complex *) out);
    Normalize(out);
  }
  
  void fftNormalized(double *in, Complex *out) {
    fftNormalized((Complex *) in,out);
  }
  
  void fft0Normalized(Complex *in, Complex *out=NULL) {
    out=Setout(in,out);
    Execute(in,out,true);
    Normalize(out);
  }
  
  void fft0Normalized(Complex *in, double *out) {
    out=(double *) Setout(in,(Complex *) out);
    Execute(in,(Complex *) out,true);
    Normalize(out);
  }
  
  void fft0Normalized(double *in, Complex *out) {
    fft0Normalized((Complex *) in,out);
  }
  
  void fftNormalized(Complex *in, Complex *out,
                     unsigned int nx, unsigned int M,
                     ptrdiff_t stride, ptrdiff_t dist) {
    if(stride == 1 && dist == (int) nx) fftw::fftNormalized(in,out);
    else if(stride == (int) nx && dist == 1) fftw::fftNormalized(in,out);
    else {
      out=Setout(in,out);
      Execute(in,out);
#ifndef FFTWPP_SINGLE_THREAD
#pragma omp parallel for num_threads(threads)
#endif
      for(unsigned int k=0; k < M; k++) {
        for(unsigned int j=0; j < nx; j++) {
          out[j*stride+k*dist] *= norm;
        }
      }
    }
  }

};

// Compute the complex Fourier transform of n complex values.
// Before calling fft(), the arrays in and out (which may coincide) must be
// allocated as Complex[n].
//
// Out-of-place usage: 
//
//   fft1d Forward(n,-1,in,out);
//   Forward.fft(in,out);
//
//   fft1d Backward(n,1,in,out);
//   Backward.fft(in,out);
//
//   fft1d Backward(n,1,in,out);
//   Backward.fftNormalized(in,out); // True inverse of Forward.fft(out,in);
//
// In-place usage:
//
//   fft1d Forward(n,-1);
//   Forward.fft(in);
//
//   fft1d Backward(n,1);
//   Backward.fft(in);
//
class fft1d : public fftw {
  unsigned int nx;
public:  
  fft1d(unsigned int nx, int sign, Complex *in=NULL, Complex *out=NULL) 
    : fftw(2*nx,sign), nx(nx) {Setup(in,out);} 
  
#ifdef __Array_h__
  fft1d(int sign, const Array::array1<Complex>& in,
        const Array::array1<Complex>& out=Array::NULL1) 
    : fftw(2*in.Nx(),sign), nx(in.Nx()) {Setup(in,out);} 
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_1d(nx,(fftw_complex *) in,(fftw_complex *) out,
                            sign,effort);
  }
};
  
// Compute the complex Fourier transform of M complex vectors, each of
// length n.
// Before calling fft(), the arrays in and out (which may coincide) must be
// allocated as Complex[M*n].
//
// Out-of-place usage: 
//
//   mfft1d Forward(n,-1,M,stride,dist,in,out);
//   Forward.fft(in,out);
//
// In-place usage:
//
//   mfft1d Forward(n,-1,M,stride,dist);
//   Forward.fft(in);
//
// Notes:
//   stride is the spacing between the elements of each Complex vector;
//   dist is the spacing between the first elements of the vectors;
//
//
class mfft1d : public fftw {
  unsigned int nx;
  unsigned int M;
  ptrdiff_t stride;
  ptrdiff_t dist;
public:  
  mfft1d(unsigned int nx, int sign, unsigned int M=1, ptrdiff_t stride=1,
         ptrdiff_t dist=0, Complex *in=NULL, Complex *out=NULL) 
    : fftw(2*((nx-1)*stride+(M-1)*Dist(nx,stride,dist)+1),sign,nx),
      nx(nx), M(M), stride(stride), dist(Dist(nx,stride,dist))
  {Setup(in,out);} 
  
  fftw_plan Plan(Complex *in, Complex *out) {
    int n=(int) nx;
    return fftw_plan_many_dft(1,&n,M,
                              (fftw_complex *) in,NULL,stride,dist,
                              (fftw_complex *) out,NULL,stride,dist,
                              sign,effort);
  }
  
  void fftNormalized(Complex *in, Complex *out=NULL) {
    fftw::fftNormalized(in,out,nx,M,stride,dist);
  }
};
  
// Compute the complex Fourier transform of n real values, using phase sign -1.
// Before calling fft(), the array in must be allocated as double[n] and
// the array out must be allocated as Complex[n/2+1]. The arrays in and out
// may coincide, allocated as Complex[n/2+1].
//
// Out-of-place usage: 
//
//   rcfft1d Forward(n,in,out);
//   Forward.fft(in,out);
//
// In-place usage:
//
//   rcfft1d Forward(n);
//   Forward.fft(out);
// 
// Notes:
//   in contains the n real values stored as a Complex array;
//   out contains the first n/2+1 Complex Fourier values.
//
class rcfft1d : public fftw {
  unsigned int nx;
public:  
  rcfft1d(unsigned int nx, Complex *out=NULL) 
    : fftw(2*(nx/2+1),-1,nx), nx(nx) {Setup(out);}
  
  rcfft1d(unsigned int nx, double *in, Complex *out=NULL) 
    : fftw(realsize(nx,in,out),-1,nx), nx(nx) {Setup(in,out);}
  
#ifdef __Array_h__
  rcfft1d(unsigned int nx, const Array::array1<Complex>& out)
    : fftw(out.Size(),-1,nx), nx(nx) {Setup(out);} 
  
  rcfft1d(unsigned int nx, const Array::array1<double>& in, 
          const Array::array1<Complex>& out=Array::NULL1)
    : fftw(realsize(nx,in(),out()),-1,nx), nx(nx) {Setup(in,out);} 
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_r2c_1d(nx,(double *) in,(fftw_complex *) out, effort);
  }
  
  void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft_r2c(plan,(double *) in,(fftw_complex *) out);
  }
};
  
// Compute the real inverse Fourier transform of the n/2+1 Complex values
// corresponding to the non-negative part of the frequency spectrum, using
// phase sign +1.
// Before calling fft(), the array in must be allocated as Complex[n/2+1]
// and the array out must be allocated as double[n]. The arrays in and out
// may coincide, allocated as Complex[n/2+1]. 
//
// Out-of-place usage (input destroyed):
//
//   crfft1d Backward(n,in,out);
//   Backward.fft(in,out);
//
// In-place usage:
//
//   crfft1d Backward(n);
//   Backward.fft(in);
// 
// Notes:
//   in contains the first n/2+1 Complex Fourier values.
//   out contains the n real values stored as a Complex array;
//
class crfft1d : public fftw {
  unsigned int nx;
public:  
  crfft1d(unsigned int nx, double *out=NULL) 
    : fftw(2*(nx/2+1),1,nx), nx(nx) {Setup(out);} 
  
  crfft1d(unsigned int nx, Complex *in, double *out=NULL) 
    : fftw(realsize(nx,in,out),1,nx), nx(nx) {Setup(in,out);} 
  
#ifdef __Array_h__
  crfft1d(unsigned int nx, const Array::array1<double>& out)
    : fftw(out.Size(),1,nx), nx(nx) {Setup(out);}
  
  crfft1d(unsigned int nx, const Array::array1<Complex>& in)
    : fftw(2*in.Size(),1,nx), nx(nx) {Setup(in);}
  
  crfft1d(unsigned int nx, const Array::array1<Complex>& in,
          const Array::array1<double>& out)
    : fftw(out.Size(),1,nx), nx(nx) {Setup(in,out);}
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_c2r_1d(nx,(fftw_complex *) in,(double *) out,effort);
  }
  
  void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft_c2r(plan,(fftw_complex *) in,(double *) out);
  }
};
  
// Compute the real Fourier transform of M real vectors, each of length n,
// using phase sign -1. Before calling fft(), the array in must be
// allocated as double[M*n] and the array out must be allocated as
// Complex[M*(n/2+1)]. The arrays in and out may coincide,
// allocated as Complex[M*(n/2+1)].
//
// Out-of-place usage: 
//
//   mrcfft1d Forward(n,M,stride,dist,in,out);
//   Forward.fft(in,out);
//
// In-place usage:
//
//   mrcfft1d Forward(n,M,stride,dist);
//   Forward.fft(out);
// 
// Notes:
//   stride is the spacing between the elements of each Complex vector;
//   dist is the spacing between the first elements of the vectors;
//   in contains the n real values stored as a Complex array;
//   out contains the first n/2+1 Complex Fourier values.
//
class mrcfft1d : public fftw {
  unsigned int nx;
  unsigned int M;
  ptrdiff_t stride;
  ptrdiff_t dist;
public:  
  mrcfft1d(unsigned int nx, unsigned int M=1, ptrdiff_t stride=1,
           ptrdiff_t dist=0, Complex *out=NULL) 
    : fftw(2*(nx/2*stride+(M-1)*Dist(nx,stride,dist)+1),-1,nx), nx(nx), M(M),
      stride(stride), dist(Dist(nx,stride,dist)) {Setup(out);} 
  
  mrcfft1d(unsigned int nx, unsigned int M=1, ptrdiff_t stride=1,
           ptrdiff_t dist=0, double *in=NULL, Complex *out=NULL) 
    : fftw(2*(nx/2*stride+(M-1)*Dist(nx,stride,dist)+1),-1,nx), nx(nx), M(M),
      stride(stride), dist(Dist(nx,stride,dist)) {Setup(in,out);} 
  
  fftw_plan Plan(Complex *in, Complex *out) {
    int n=(int) nx;
    return fftw_plan_many_dft_r2c(1,&n,M,
                                  (double *) in,NULL,stride,2*dist,
                                  (fftw_complex *) out,NULL,stride,dist,
                                  effort);
  }
  
  void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft_r2c(plan,(double *) in,(fftw_complex *) out);
  }
  
  void fftNormalized(Complex *in, Complex *out=NULL) {
    fftw::fftNormalized(in,out,nx/2+1,M,stride,dist);
  }
};
  
// Compute the real inverse Fourier transform of M complex vectors, each of
// length n/2+1, corresponding to the non-negative parts of the frequency
// spectra, using phase sign +1. Before calling fft(), the array in must be
// allocated as Complex[M*(n/2+1)] and the array out must be allocated as
// double[M*n]. The arrays in and out may coincide,
// allocated as Complex[M*(n/2+1)].  
//
// Out-of-place usage (input destroyed):
//
//   mcrfft1d Backward(n,M,stride,dist,in,out);
//   Backward.fft(in,out);
//
// In-place usage:
//
//   mcrfft1d Backward(n,M,stride,dist);
//   Backward.fft(out);
// 
// Notes:
//   stride is the spacing between the elements of each Complex vector;
//   dist is the spacing between the first elements of the vectors;
//   in contains the first n/2+1 Complex Fourier values.
//   out contains the n real values stored as a Complex array;
//
class mcrfft1d : public fftw {
  unsigned int nx;
  unsigned int M;
  ptrdiff_t stride;
  ptrdiff_t dist;
public:
  mcrfft1d(unsigned int nx, unsigned int M=1, ptrdiff_t stride=1,
           ptrdiff_t dist=0, Complex *in=NULL, double *out=NULL) 
    : fftw((realsize(nx,in,out)-2)*stride+2*(M-1)*Dist(nx,stride,dist)+2,1,nx),
      nx(nx), M(M), stride(stride), dist(Dist(nx,stride,dist)) {Setup(in,out);}
  
  fftw_plan Plan(Complex *in, Complex *out) {
    int n=(int) nx;
    return fftw_plan_many_dft_c2r(1,&n,M,
                                  (fftw_complex *) in,NULL,stride,dist,
                                  (double *) out,NULL,stride,2*dist,
                                  effort);
  }
  
  void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft_c2r(plan,(fftw_complex *) in,(double *) out);
  }
  
  void fftNormalized(Complex *in, Complex *out=NULL) {
    fftw::fftNormalized(in,out,(nx/2+1),M,stride,dist);
  }
};
  
// Compute the complex two-dimensional Fourier transform of nx times ny
// complex values. Before calling fft(), the arrays in and out (which may
// coincide) must be allocated as Complex[nx*ny].
//
// Out-of-place usage: 
//
//   fft2d Forward(nx,ny,-1,in,out);
//   Forward.fft(in,out);
//
//   fft2d Backward(nx,ny,1,in,out);
//   Backward.fft(in,out);
//
//   fft2d Backward(nx,ny,1,in,out);
//   Backward.fftNormalized(in,out); // True inverse of Forward.fft(out,in);
//
// In-place usage:
//
//   fft2d Forward(nx,ny,-1);
//   Forward.fft(in);
//
//   fft2d Backward(nx,ny,1);
//   Backward.fft(in);
//
// Note:
//   in[ny*i+j] contains the ny Complex values for each i=0,...,nx-1.
//
class fft2d : public fftw {
  unsigned int nx;
  unsigned int ny;
public:  
  fft2d(unsigned int nx, unsigned int ny, int sign, Complex *in=NULL,
        Complex *out=NULL)
    : fftw(2*nx*ny,sign), nx(nx), ny(ny) {Setup(in,out);} 
  
#ifdef __Array_h__
  fft2d(int sign, const Array::array2<Complex>& in,
        const Array::array2<Complex>& out=Array::NULL2) 
    : fftw(2*in.Size(),sign), nx(in.Nx()), ny(in.Ny()) {Setup(in,out);} 
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_2d(nx,ny,(fftw_complex *) in,(fftw_complex *) out,
                            sign,effort);
  }
  
  void Execute(Complex *in, Complex *out, bool=false) {
    fftw_execute_dft(plan,(fftw_complex *) in,(fftw_complex *) out);
  }
};

// Compute the complex two-dimensional Fourier transform of nx times ny real
// values, using phase sign -1.
// Before calling fft(), the array in must be allocated as double[nx*ny] and
// the array out must be allocated as Complex[nx*(ny/2+1)]. The arrays in
// and out may coincide, allocated as Complex[nx*(ny/2+1)]. 
//
// Out-of-place usage: 
//
//   rcfft2d Forward(nx,ny,in,out);
//   Forward.fft(in,out);       // Origin of Fourier domain at (0,0)
//
// In-place usage:
//
//   rcfft2d Forward(nx,ny);
//   Forward.fft(in);           // Origin of Fourier domain at (0,0)
//   Forward.fft0(in);          // Origin of Fourier domain at (nx/2,0)
// 
// Notes:
//   in contains the nx*ny real values stored as a Complex array;
//   out contains the upper-half portion (ky >= 0) of the Complex transform.
//
class rcfft2d : public fftw {
  unsigned int nx;
  unsigned int ny;
public:  
  rcfft2d(unsigned int nx, unsigned int ny, Complex *out=NULL)
    : fftw(2*nx*(ny/2+1),-1,nx*ny), nx(nx), ny(ny) {Setup(out);} 
  
  rcfft2d(unsigned int nx, unsigned int ny, double *in, Complex *out=NULL)
    : fftw(nx*realsize(ny,in,out),-1,nx*ny), nx(nx), ny(ny) {Setup(in,out);} 
  
#ifdef __Array_h__
  rcfft2d(unsigned int ny, const Array::array2<Complex>& out)
    : fftw(out.Size(),-1,out.Nx()*ny), nx(out.Nx()), ny(ny) {Setup(out);} 
  
  rcfft2d(unsigned int ny, const Array::array2<double>& in,
          const Array::array2<Complex>& out=Array::NULL2)
    : fftw(in.Nx()*realsize(ny,in(),out()),-1,in.Nx()*ny), nx(in.Nx()), ny(ny)
  {Setup(in,out);} 
#endif
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_r2c_2d(nx,ny,(double *) in,(fftw_complex *) out,
                                effort);
  }
  
  void Execute(Complex *in, Complex *out, bool shift=false) {
    if(shift && inplace) Shift(in,nx,ny);
    fftw_execute_dft_r2c(plan,(double *) in,(fftw_complex *) out);
  }
};
  
// Compute the real two-dimensional inverse Fourier transform of the
// nx*(ny/2+1) Complex values corresponding to the spectral values in the
// half-plane ky >= 0, using phase sign +1.
// Before calling fft(), the array in must be allocated as
// Complex[nx*(ny+1)/2] and the array out must be allocated as
// double[nx*ny]. The arrays in and out may coincide,
// allocated as Complex[nx*(ny/2+1)]. 
//
// Out-of-place usage (input destroyed):
//
//   crfft2d Backward(nx,ny,in,out);
//   Backward.fft(in,out);      // Origin of Fourier domain at (0,0)
//   Backward.fft0(in,out);     // Origin of Fourier domain at (nx/2,0)
//
// In-place usage:
//
//   crfft2d Backward(nx,ny);
//   Backward.fft(in);          // Origin of Fourier domain at (0,0)
//   Backward.fft0(in);         // Origin of Fourier domain at (nx/2,0)
// 
// Notes:
//   in contains the upper-half portion (ky >= 0) of the Complex transform;
//   out contains the nx*ny real values stored as a Complex array.
//
class crfft2d : public fftw {
  unsigned int nx;
  unsigned int ny;
public:  
  crfft2d(unsigned int nx, unsigned int ny, Complex *in=NULL) :
    fftw(2*nx*(ny/2+1),1,nx*ny), nx(nx), ny(ny) {Setup(in);} 
  
  crfft2d(unsigned int nx, unsigned int ny, Complex *in, double *out) :
    fftw(nx*realsize(ny,in,out),1,nx*ny), nx(nx), ny(ny) {Setup(in,out);} 
  
#ifdef __Array_h__
  crfft2d(unsigned int ny, const Array::array2<double>& out)
    : fftw(out.Size(),1,out.Nx()*ny), nx(out.Nx()), ny(ny) {Setup(out);} 
  
  crfft2d(unsigned int ny, const Array::array2<Complex>& in)
    : fftw(2*in.Size(),1,in.Nx()*ny), nx(in.Nx()), ny(ny) {Setup(in);}
  
  crfft2d(unsigned int ny, const Array::array2<Complex>& in,
          const Array::array2<double>& out)
    : fftw(out.Size(),1,in.Nx()*ny), nx(in.Nx()), ny(ny) {Setup(in,out);}
#endif
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_c2r_2d(nx,ny,(fftw_complex *) in,(double *) out,
                                effort);
  }
  
  void Execute(Complex *in, Complex *out, bool shift=false) {
    fftw_execute_dft_c2r(plan,(fftw_complex *) in,(double *) out);
    if(shift) Shift(out,nx,ny);
  }
};

// Compute the complex three-dimensional Fourier transform of 
// nx times ny times nz complex values. Before calling fft(), the arrays in
// and out (which may coincide) must be allocated as Complex[nx*ny*nz].
//
// Out-of-place usage: 
//
//   fft3d Forward(nx,ny,nz,-1,in,out);
//   Forward.fft(in,out);
//
//   fft3d Backward(nx,ny,nz,1,in,out);
//   Backward.fft(in,out);
//
//   fft3d Backward(nx,ny,nz,1,in,out);
//   Backward.fftNormalized(in,out); // True inverse of Forward.fft(out,in);
//
// In-place usage:
//
//   fft3d Forward(nx,ny,nz,-1);
//   Forward.fft(in);
//
//   fft3d Backward(nx,ny,nz,1);
//   Backward.fft(in);
//
// Note:
//   in[nz*(ny*i+j)+k] contains the (i,j,k)th Complex value,
//   indexed by i=0,...,nx-1, j=0,...,ny-1, and k=0,...,nz-1.
//
class fft3d : public fftw {
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
public:  
  fft3d(unsigned int nx, unsigned int ny, unsigned int nz,
        int sign, Complex *in=NULL, Complex *out=NULL)
    : fftw(2*nx*ny*nz,sign), nx(nx), ny(ny), nz(nz) {Setup(in,out);} 
  
#ifdef __Array_h__
  fft3d(int sign, const Array::array3<Complex>& in,
        const Array::array3<Complex>& out=Array::NULL3)
    : fftw(2*in.Size(),sign), nx(in.Nx()), ny(in.Ny()), nz(in.Nz()) 
  {Setup(in,out);}
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_3d(nx,ny,nz,(fftw_complex *) in,
                            (fftw_complex *) out, sign, effort);
  }
};

// Compute the complex two-dimensional Fourier transform of
// nx times ny times nz real values, using phase sign -1.
// Before calling fft(), the array in must be allocated as double[nx*ny*nz]
// and the array out must be allocated as Complex[nx*ny*(nz/2+1)]. The
// arrays in and out may coincide, allocated as Complex[nx*ny*(nz/2+1)]. 
//
// Out-of-place usage: 
//
//   rcfft3d Forward(nx,ny,nz,in,out);
//   Forward.fft(in,out);       // Origin of Fourier domain at (0,0)
//
// In-place usage:
//
//   rcfft3d Forward(nx,ny,nz);
//   Forward.fft(in);           // Origin of Fourier domain at (0,0)
//   Forward.fft0(in);          // Origin of Fourier domain at (nx/2,ny/2,0)
// 
// Notes:
//   in contains the nx*ny*nz real values stored as a Complex array;
//   out contains the upper-half portion (kz >= 0) of the Complex transform.
//
class rcfft3d : public fftw {
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
public:  
  rcfft3d(unsigned int nx, unsigned int ny, unsigned int nz, Complex *out=NULL)
    : fftw(2*nx*ny*(nz/2+1),-1,nx*ny*nz), nx(nx), ny(ny), nz(nz) {Setup(out);} 
  
  rcfft3d(unsigned int nx, unsigned int ny, unsigned int nz, double *in,
          Complex *out=NULL) : fftw(nx*ny*realsize(nz,in,out),-1,nx*ny*nz),
                               nx(nx), ny(ny), nz(nz) {Setup(in,out);} 
  
#ifdef __Array_h__
  rcfft3d(unsigned int nz, const Array::array3<Complex>& out)
    : fftw(out.Size(),-1,out.Nx()*out.Ny()*nz),
      nx(out.Nx()), ny(out.Ny()), nz(nz) {Setup(out);} 
  
  rcfft3d(unsigned int nz, const Array::array3<double>& in,
          const Array::array3<Complex>& out=Array::NULL3)
    : fftw(in.Nx()*in.Ny()*realsize(nz,in(),out()),-1,in.Size()),
      nx(in.Nx()), ny(in.Ny()), nz(nz) {Setup(in,out);} 
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_r2c_3d(nx,ny,nz,(double *) in,(fftw_complex *) out,
                                effort);
  }
  
  void Execute(Complex *in, Complex *out, bool shift=false) {
    if(shift && inplace) Shift(in,nx,ny,nz);
    fftw_execute_dft_r2c(plan,(double *) in,(fftw_complex *) out);
  }
};
  
// Compute the real two-dimensional inverse Fourier transform of the
// nx*ny*(nz/2+1) Complex values corresponding to the spectral values in the
// half-plane kz >= 0, using phase sign +1.
// Before calling fft(), the array in must be allocated as
// Complex[nx*ny*(nz+1)/2] and the array out must be allocated as
// double[nx*ny*nz]. The arrays in and out may coincide,
// allocated as Complex[nx*ny*(nz/2+1)]. 
//
// Out-of-place usage (input destroyed):
//
//   crfft3d Backward(nx,ny,nz,in,out);
//   Backward.fft(in,out);      // Origin of Fourier domain at (0,0)
//   Backward.fft0(in,out);     // Origin of Fourier domain at (nx/2,ny/2,0)
//
// In-place usage:
//
//   crfft3d Backward(nx,ny,nz);
//   Backward.fft(in);          // Origin of Fourier domain at (0,0)
//   Backward.fft0(in);         // Origin of Fourier domain at (nx/2,ny/2,0)
// 
// Notes:
//   in contains the upper-half portion (kz >= 0) of the Complex transform;
//   out contains the nx*ny*nz real values stored as a Complex array.
//
class crfft3d : public fftw {
  unsigned int nx;
  unsigned int ny;
  unsigned int nz;
public:  
  crfft3d(unsigned int nx, unsigned int ny, unsigned int nz, Complex *in=NULL)
    : fftw(2*nx*ny*(nz/2+1),1,nx*ny*nz), nx(nx), ny(ny), nz(nz)
  {Setup(in);} 
  
  crfft3d(unsigned int nx, unsigned int ny, unsigned int nz, Complex *in,
          double *out=NULL) : 
    fftw(nx*ny*(realsize(nz,in,out)),1,nx*ny*nz), nx(nx), ny(ny), nz(nz)
  {Setup(in,out);} 
  
#ifdef __Array_h__
  crfft3d(unsigned int nz, const Array::array3<double>& out)
    : fftw(out.Size(),1,out.Nx()*out.Ny()*nz),
      nx(out.Nx()), ny(out.Ny()), nz(nz) {Setup(out);} 
  
  crfft3d(unsigned int nz, const Array::array3<Complex>& in)
    : fftw(2*in.Size(),1,in.Nx()*in.Ny()*nz),
      nx(in.Nx()), ny(in.Ny()), nz(nz) {Setup(in);} 
  
  crfft3d(unsigned int nz, const Array::array3<Complex>& in,
          const Array::array3<double>& out)
    : fftw(out.Size(),1,in.Nx()*in.Ny()*nz),
      nx(in.Nx()), ny(in.Ny()), nz(nz) {Setup(in,out);} 
#endif  
  
  fftw_plan Plan(Complex *in, Complex *out) {
    return fftw_plan_dft_c2r_3d(nx,ny,nz,(fftw_complex *) in,(double *) out,
                                effort);
  }
  
  void Execute(Complex *in, Complex *out, bool shift=false) {
    fftw_execute_dft_c2r(plan,(fftw_complex *) in,(double *) out);
    if(shift) Shift(out,nx,ny,nz);
  }
};
  
}

#endif

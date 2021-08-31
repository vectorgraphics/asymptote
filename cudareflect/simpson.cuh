#include <cuda.h>

#include "linalg.cuh"

// Compute a numerical approximation to an integral via adaptive Simpson's Rule
// This routine ignores underflow.

__device__ constexpr float sixth=1.0/6.0;
__device__ constexpr int nest=10;//10;

template<typename TRet=glm::vec3>
struct TABLE {
  bool left;                    // left interval?
  float dat;
  TRet psum, f1t, f2t, f3t, estr;
};

template<typename TRet=glm::vec3, typename TInit=DefaultVec3ZeroInit, typename T>
__device__ inline TRet
simpson(T f,                  // Function to be integrated.
        float a, float b,     // Lower, upper limits of integration.
        float acc)            // Desired relative accuracy of integral.
                              // Try to make |error| <= acc*abs(integral).
{
  TRet integral,diff,area,estl,estr,est,fv0,fv1,fv2,fv3,fv4;
  float dx;
  TABLE<TRet> table[nest],*p,*pstop;

  p=table;
  pstop=table+nest-1;
  p->left=true;
  p->psum=TInit::init();
  float alpha=a;
  float da=b-a;
  fv0=f(alpha);
  fv2=f(alpha+0.5f*da);
  fv4=f(alpha+da);
  float wt=sixth*da;
  est=wt*(fv0+4.0f*fv2+fv4);
  area=est;
  float acc2=acc*acc;

  // Have estimate est of integral on (alpha, alpha+da).
  // Bisect and compute estimates on left and right half intervals.
  // integral is the best value for the integral.

  for(;;) {
    dx=0.5f*da;
    float arg=alpha+0.5f*dx;
    fv1=f(arg);
    fv3=f(arg+dx);
    wt=sixth*dx;
    estl=wt*(fv0+4.0f*fv1+fv2);
    estr=wt*(fv2+4.0f*fv3+fv4);
    integral=estl+estr;
    diff=est-integral;
    area -= diff;

    if(p >= pstop || (TInit::abs2(diff) <= acc2*TInit::abs2(area))) {
      // Accept approximate integral.
      // If it was a right interval, add results to finish at this level.
      // If it was a left interval, process right interval.

      for(;;) {
        if(p->left == false) { // process right-half interval
          alpha += da;
          p->left=true;
          p->psum=integral;
          fv0=p->f1t;
          fv2=p->f2t;
          fv4=p->f3t;
          da=p->dat;
          est=p->estr;
          break;
        }
        integral += p->psum;
        if(--p <= table) return integral;
      }

    } else {
      // Raise level and store information for processing right-half interval.
      ++p;
      da=dx;
      est=estl;
      p->left=false;
      p->f1t=fv2;
      p->f2t=fv3;
      p->f3t=fv4;
      p->dat=dx;
      p->estr=estr;
      fv4=fv2;
      fv2=fv1;
    }
  }
}


real mean(real A[])
{
  return sum(A)/A.length;
}

// unbiased estimate
real variance(real A[])
{
 return sum((A-mean(A))^2)/(A.length-1);
}

real variancebiased(real A[])
{
 return sum((A-mean(A))^2)/A.length;
}

// unbiased estimate
real stdev(real A[])
{
 return sqrt(variance(A));
}

real rms(real A[])
{
  return sqrt(sum(A^2)/A.length);
}

real skewness(real A[])
{
  real[] diff=A-mean(A);
  return sum(diff^3)/sqrt(sum(diff^2)^3/A.length);
}

real kurtosis(real A[])
{
  real[] diff=A-mean(A);
  return sum(diff^4)/sum(diff^2)^2*A.length;
}

real kurtosisexcess(real A[])
{
  return kurtosis(A)-3;
}

real Gaussian(real x, real sigma)
{
  static real sqrt2pi=sqrt(2pi);
  return exp(-0.5*(x/sigma)^2)/(sigma*sqrt2pi);
}

real Gaussian(real x)
{
  static real invsqrt2pi=1/sqrt(2pi);
  return exp(-0.5*x^2)*invsqrt2pi;
}

// Return frequency count of data in [bins[i],bins[i+1]) for i=0,...n-1.
int[] frequency(real[] data, real[] bins)
{
  int n=bins.length-1;
  int[] freq=new int[n];
  for(int i=0; i < n; ++i) {
    real left=bins[i];
    real right=bins[i+1];
    freq[i]=sum(left <= data && data < right);
  }
  return freq;
}

guide halfbox(pair a, pair b)
{
  return a--(a.x,b.y)--b;
}

// Draw a histogram from count[n], given bin boundaries bin[n+1].
void histogram(picture pic=currentpicture, real[] count, real[] bins,
	       pen p=currentpen)
{
  bool[] valid=count > -infinity;
  real low=floor(min(valid ? count : null));
  real last=low;
  int n=count.length;
  for(int i=0; i < n; ++i) {
    if(valid[i]) {
      real c=count[i];
      draw(pic,halfbox((bins[i],last),(bins[i+1],c)),p);
      last=c;
    } else {
      if(last != low) 
	draw(pic,(bins[i],last)--(bins[i],low),p); last=low;
    }
  }
  if(last != low) draw(pic,(bins[n],last)--(bins[n],low),p);
}

// return a random number uniformly distributed in the unit interval [0,1]
real unitrand()
{			  
  return ((real) rand())/randMax();
}

// return a pair of central Gaussian random numbers with unit variance
pair Gaussrandpair()
{
  real r2,v1,v2;
  do {
    v1=2.0*unitrand()-1.0;
    v2=2.0*unitrand()-1.0;
    r2=v1*v1+v2*v2;
  } while(r2 >= 1.0 || r2 == 0.0);
  return (v1,v2)*sqrt(-log(r2)/r2);
}

// return a central Gaussian random number with unit variance
real Gaussrand()
{
  static real sqrt2=sqrt(2.0);
  static pair z;
  static bool cached=true;
  cached=!cached;
  if(cached) return sqrt2*z.y;
  z=Gaussrandpair();
  return sqrt2*z.x;
}

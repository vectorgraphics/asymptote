static import graph;

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

// Return frequency count in [bins[i],bins[i+1]) for i=0,...n-1 of data.
int[] frequency(real[] bins, real[] data)
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

// Draw a histogram for bin boundaries bin[n+1] of frequency data in count[n].
void histogram(picture pic=currentpicture, real[] bins, real[] count,
	       real low=-infinity, pen p=currentpen)
{
  bool[] valid=count > 0;
  real m=min(valid ? count : null);
  real M=max(valid ? count : null);
  bounds my=autoscale(pic.scale.y.scale.T(m),pic.scale.y.T(M),
		      pic.scale.y.scale);
  if(low == -infinity) low=pic.scale.y.scale.Tinv(my.min);
  real last=low;
  int n=count.length;
  begingroup(pic);
  for(int i=0; i < n; ++i) {
    if(valid[i]) {
      real c=count[i];
      draw(pic,halfbox(Scale(pic,(bins[i],last)),Scale(pic,(bins[i+1],c))),p);
      last=c;
    } else {
      if(last != low) {
	draw(pic,Scale(pic,(bins[i],last))--Scale(pic,(bins[i],low)),p);
	last=low;
      }
    }
  }
  if(last != low)
    draw(pic,Scale(pic,(bins[n],last))--Scale(pic,(bins[n],low)),p);
  endgroup(pic);
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

struct linefit {
  public real m,b;	// slope, intercept
  public real dm,db;	// standard error in slope, intercept
  public real r;	// correlation coefficient
  real fit(real x) {
    return m*x+b;
  }
}

linefit operator init() {return new linefit;}
  
// Do a least-squares fit of data in real arrays x and y to the line y=m*x+b
linefit leastsquares(real[] x, real[] y)
{
  linefit L;
  int n=x.length;
  if(n == 1) abort("Least squares fit requires at least 2 data points");
  real sx=sum(x);
  real sy=sum(y);
  real sxx=n*sum(x^2)-sx^2;
  real sxy=n*sum(x*y)-sx*sy;
  L.m=sxy/sxx;
  L.b=(sy-L.m*sx)/n;
  if(n > 2) {
    real syy=n*sum(y^2)-sy^2;
    real s=sqrt((syy-sxy^2/sxx)/(n-2));
    L.r=sxy/sqrt(sxx*syy);
    L.dm=s*sqrt(1/sxx);
    L.db=s*sqrt(1+sx^2/sxx)/n;
  }  
  return L;
}

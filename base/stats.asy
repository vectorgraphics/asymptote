private import graph;

real legendmarkersize=2mm;

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

// Return frequency count of data in [bins[i],bins[i+1]) for i=0,...,n-1.
int[] frequency(real[] data, real[] bins)
{
  int n=bins.length-1;
  int[] freq=new int[n];
  for(int i=0; i < n; ++i)
    freq[i]=sum(bins[i] <= data & data < bins[i+1]);
  return freq;
}

// Return frequency count in n uniform bins from a to b
// (faster than the above more general algorithm).
int[] frequency(real[] data, real a, real b, int n)
{
  int[] freq=sequence(new int(int x) {return 0;},n);
  real h=n/(b-a);
  for(int i=0; i < data.length; ++i) {
    int I=Floor((data[i]-a)*h);
    if(I >= 0 && I < n)
      ++freq[I];
  }
  return freq;
}

// Return frequency count in [xbins[i],xbins[i+1]) and [ybins[j],ybins[j+1]).
int[][] frequency(real[] x, real[] y, real[] xbins, real[] ybins)
{
  int n=xbins.length-1;
  int m=ybins.length-1;
  int[][] freq=new int[n][m];
  bool[][] inybin=new bool[m][y.length];
  for(int j=0; j < m; ++j)
    inybin[j]=ybins[j] <= y & y < ybins[j+1];
  for(int i=0; i < n; ++i) {
    bool[] inxbini=xbins[i] <= x & x < xbins[i+1];
    int[] freqi=freq[i];
    for(int j=0; j < m; ++j)
      freqi[j]=sum(inxbini & inybin[j]);
  }
  return freq;
}

// Return frequency count in nx by ny uniform bins in box(a,b).
int[][] frequency(real[] x, real[] y, pair a, pair b, int nx, int ny=nx)
{
  int[][] freq=new int[nx][];
  for(int i=0; i < nx; ++i)
    freq[i]=sequence(new int(int x) {return 0;},ny);
  real hx=nx/(b.x-a.x);
  real hy=ny/(b.y-a.y);
  real ax=a.x;
  real ay=a.y;
  for(int i=0; i < x.length; ++i) {
    int I=Floor((x[i]-ax)*hx);
    int J=Floor((y[i]-ay)*hy);
    if(I >= 0 && I <= nx && J >= 0 && J <= ny)
      ++freq[I][J];
  }
  return freq;
}

int[][] frequency(pair[] z, pair a, pair b, int nx, int ny=nx)
{
  int[][] freq=new int[nx][];
  for(int i=0; i < nx; ++i)
    freq[i]=sequence(new int(int x) {return 0;},ny);
  real hx=nx/(b.x-a.x);
  real hy=ny/(b.y-a.y);
  real ax=a.x;
  real ay=a.y;
  for(int i=0; i < z.length; ++i) {
    int I=Floor((z[i].x-ax)*hx);
    int J=Floor((z[i].y-ay)*hy);
    if(I >= 0 && I < nx && J >= 0 && J < ny)
      ++freq[I][J];
  }
  return freq;
}

path halfbox(pair a, pair b)
{
  return a--(a.x,b.y)--b;
}

path topbox(pair a, pair b)
{
  return a--(a.x,b.y)--b--(b.x,a.y);
}

// Draw a histogram for bin boundaries bin[n+1] of frequency data in count[n].
void histogram(picture pic=currentpicture, real[] bins, real[] count,
               real low=-infinity,
	       pen fillpen=nullpen, pen drawpen=nullpen, bool bars=false,
	       Label legend="", real markersize=legendmarkersize)
{
  if((fillpen == nullpen || bars == true) && drawpen == nullpen)
    drawpen=currentpen;
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
      pair b=Scale(pic,(bins[i+1],c));
      pair a=Scale(pic,(bins[i],low));
      if(fillpen != nullpen) {
	fill(pic,box(a,b),fillpen);
	if(!bars) draw(pic,b--(b.x,a.y),fillpen);
      }
      if(!bars)
	draw(pic,halfbox(Scale(pic,(bins[i],last)),b),drawpen);
      else draw(pic,topbox(a,b),drawpen);
      last=c;
    } else {
      if(!bars && last != low) {
	draw(pic,Scale(pic,(bins[i],last))--Scale(pic,(bins[i],low)),drawpen);
        last=low;
      }
    }
  }
  if(!bars && last != low)
    draw(pic,Scale(pic,(bins[n],last))--Scale(pic,(bins[n],low)),drawpen);
  endgroup(pic);

  if(legend.s != "") {
    marker m=marker(scale(markersize)*shift((-0.5,-0.5))*unitsquare,
		    drawpen,fillpen == nullpen ? Draw :
		    (drawpen == nullpen ? Fill(fillpen) : FillDraw(fillpen)));
    legend.p(drawpen);
    pic.legend.push(Legend(legend.s,legend.p,invisible,m.f));
  }
}

// Draw a histogram for data in n uniform bins between a and b
// (optionally normalized).
void histogram(picture pic=currentpicture, real[] data, real a, real b, int n,
               bool normalize=false, real low=-infinity,
	       pen fillpen=nullpen, pen drawpen=nullpen, bool bars=false,
	       Label legend="", real markersize=legendmarkersize)
{
  real dx=(b-a)/n;
  real[] freq=frequency(data,a,b,n);
  if(normalize) freq /= dx*sum(freq);
  histogram(pic,a+sequence(n+1)*dx,freq,low,fillpen,drawpen,bars,legend,
	    markersize);
}

// Method of Shimazaki and Shinomoto for selecting the optimal number of bins.
// Shimazaki H. and Shinomoto S., A method for selecting the bin size of a
// time histogram, Neural Computation (2007), Vol. 19(6), 1503-1527.
// cf. http://www.ton.scphys.kyoto-u.ac.jp/~hideaki/res/histogram.html
int bins(real[] data, int max=100)
{
  real m=min(data);
  real M=max(data)*(1+epsilon);
  real n=data.length;
  int bins=1;
  real minC=2n-n^2; // Cost function for N=1.
  for(int N=2; N <= max; ++N) {
    real C=N*(2n-sum(frequency(data,m,M,N)^2));
    if(C < minC) {
      minC=C;
      bins=N;
    }
  }

  return bins;
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
  real m,b;     // slope, intercept
  real dm,db;   // standard error in slope, intercept
  real r;       // correlation coefficient
  real fit(real x) {
    return m*x+b;
  }
}

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
    if(sxx == 0 || syy == 0) return L;
    L.r=sxy/sqrt(sxx*syy);
    real arg=syy-sxy^2/sxx;
    if(arg <= 0) return L;
    real s=sqrt(arg/(n-2));
    L.dm=s*sqrt(1/sxx);
    L.db=s*sqrt(1+sx^2/sxx)/n;
  }  
  return L;
}

// A rainbow palette tapering off to black/white at the spectrum ends.
// two=true means circle the color wheel twice, scaling the intensity linearly.
pen[] BWrainbow(int NColors=65536, bool two=false)
{
  int offset=1;
  int nintervals=6;
  int divisor=3;
  
  if(two) nintervals += 6;
  
  int num=NColors-offset;
  int n=(num/(nintervals*divisor))*divisor;
  NColors=n*nintervals+offset;
		
  pen[] Palette;
  if(n == 0) return Palette;
  
  Palette=new pen[NColors];
  real ninv=1.0/n;

  int N1,N2,N3,N4,N5;
  int k=0;
  
  if(two) {
    N1=n;
    N2=2n;
    N3=3n;
    N4=4n;
    N5=5n;
    for(int i=0; i < n; ++i) {
      real ininv=i*ninv;
      real ininv1=1.0-ininv;
      Palette[i]=rgb(ininv1,0.0,1.0);
      Palette[N1+i]=rgb(0.0,ininv,1.0);
      Palette[N2+i]=rgb(0.0,1.0,ininv1);
      Palette[N3+i]=rgb(ininv,1.0,0.0);
      Palette[N4+i]=rgb(1.0,ininv1,0.0);
      Palette[N5+i]=rgb(1.0,0.0,ininv);
    }
    k += 6n;
  }
  
  if(two)
    for(int i=0; i < n; ++i) 
      Palette[k+i]=rgb(1.0-i*ninv,0.0,1.0);
  else {
    int n3=n/3;
    int n23=2*n3;
    real third=n3*ninv;
    real twothirds=n23*ninv;
    N1=k;
    N2=k+n3;
    N3=k+n23;
    for(int i=0; i < n3; ++i) {
      real ininv=i*ninv;
      Palette[N1+i]=rgb(ininv,0.0,ininv);
      Palette[N2+i]=rgb(third,0.0,third+ininv);
      Palette[N3+i]=rgb(third-ininv,0.0,twothirds+ininv);
    }
  }
  k += n;

  N1=k;
  N2=N1+n;
  N3=N2+n;
  N4=N3+n;
  N5=N4+n;
  for(int i=0; i < n; ++i) {
    real ininv=i*ninv;
    real ininv1=1.0-ininv;
    Palette[N1+i]=rgb(0.0,ininv,1.0);
    Palette[N2+i]=rgb(0.0,1.0,ininv1);
    Palette[N3+i]=rgb(ininv,1.0,0.0);    
    Palette[N4+i]=rgb(1.0,ininv1,0.0);
    Palette[N5+i]=rgb(1.0,ininv,ininv);
  }
  k=N5+n;
  Palette[k]=rgb(1.0,1.0,1.0);
  
  if(two) {
    real NColorsinv=1.0/NColors;
    for(int i=0; i <= k; ++i)
      Palette[i]=i*NColorsinv*Palette[i];
  }
  return Palette;
}

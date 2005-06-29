static import graph;

void image(picture pic=currentpicture, real[][] data, pen[] palette,
	   pair initial, pair final)
{
  pic.add(new void (frame f, transform t) {
    image(f,data,palette,t*initial,t*final);
    });
  pic.addBox(initial,final);
}

typedef ticks paletteticks(real Size);

paletteticks PaletteTicks(bool begin=true, int N=0, real Step=0,
			  string F=defaultformat, bool end=true)
{
  return new ticks(real Size) {
    return LeftTicks(begin,N,0,Step,0.0,Size,0.0,ticklabel(F),end);
  };
} 

public paletteticks PaletteTicks=PaletteTicks();

picture palette(real[][] data, real width=Ticksize,
	     pen[] palette, string s="", real position=0.5,
	     real angle=infinity, pair align=E, pair shift=0,
	     pair side=right, pen plabel=currentpen, pen p=nullpen,
	     paletteticks ticks=PaletteTicks)
{
  if(p == nullpen) p=plabel;
  picture pic;
  real initialy=min(data);
  real finaly=max(data);
  pair z0=(0,initialy);
  pair z1=(0,finaly);
  
  pic.add(new void (frame f, transform t) {
	    pair Z0=(0,(t*z0).y);
	    pair Z1=(0,(t*z1).y);
	    pair initial=Z0-width;
	    image(f,new real[][] {sequence(palette.length-1)},palette,
		  initial,Z1);
	    draw(f,Z0--initial--Z1-width--Z1,p);
  });
  
  pic.addBox(z0,z1,(-width,0),(0,0));
  yaxis(pic,initialy,finaly,s,position,angle,align,shift,side,plabel,p,
	ticks(width),Above);
  return pic;
}

// A grayscale palette
pen[] Grayscale(int NColors=256)
{
  real ninv=1.0/(NColors-1.0);
  return sequence(new pen(int i) {return gray(i*ninv);},NColors);
}

// A rainbow palette
pen[] Rainbow(int NColors=32766)
{
  int offset=1;
  int nintervals=5;
  int n=quotient(NColors-1,nintervals);
		
  pen[] Palette;
  if(n == 0) return Palette;
  
  Palette=new pen[n*nintervals+offset];
  real ninv=1.0/n;

  int N2=2n;
  int N3=3n;
  int N4=4n;
  for(int i=0; i < n; ++i) {
    real ininv=i*ninv;
    real ininv1=1.0-ininv;
    Palette[i]=rgb(ininv1,0.0,1.0);
    Palette[n+i]=rgb(0.0,ininv,1.0);
    Palette[N2+i]=rgb(0.0,1.0,ininv1);
    Palette[N3+i]=rgb(ininv,1.0,0.0);    
    Palette[N4+i]=rgb(1.0,ininv1,0.0);
  }
  Palette[N4+n]=rgb(1.0,0.0,0.0);
  
  return Palette;
}

private pen[] BWRainbow(int NColors, bool two)
{
  int offset=1;
  int nintervals=6;
  int divisor=3;
  
  if(two) nintervals += 6;
  
  int num=NColors-offset;
  int n=quotient(num,nintervals*divisor)*divisor;
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
    int n3=quotient(n,3);
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
  
  return Palette;
}

// A rainbow palette tapering off to black/white at the spectrum ends,
pen[] BWRainbow(int NColors=32761)
{
  return BWRainbow(NColors,false);
}

// A double rainbow palette tapering off to black/white at the spectrum ends,
// with a linearly scaled intensity.
pen[] BWRainbow2(int NColors=32761)
{
  pen[] Palette=BWRainbow(NColors,true);
  int n=Palette.length;
  real ninv=1.0/n;
  for(int i=0; i < n; ++i)
    Palette[i]=i*ninv*Palette[i];
  return Palette;
}

pen[] cmyk(pen[] Palette) 
{
  int n=Palette.length;
  for(int i=0; i < n; ++i)
    Palette[i]=cmyk+Palette[i];
  return Palette;
}
  

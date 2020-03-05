private import graph;

private transform swap=(0,0,0,1,1,0);

typedef bounds range(picture pic, real min, real max);

range Range(bool automin=false, real min=-infinity,
            bool automax=false, real max=infinity) 
{
  return new bounds(picture pic, real dmin, real dmax) {
    // autoscale routine finds reasonable limits
    bounds mz=autoscale(pic.scale.z.T(dmin),
                        pic.scale.z.T(dmax),
                        pic.scale.z.scale);
    // If automin/max, use autoscale result, else
    //   if min/max is finite, use specified value, else
    //   use minimum/maximum data value
    real pmin=automin ? pic.scale.z.Tinv(mz.min) : (finite(min) ? min : dmin);
    real pmax=automax ? pic.scale.z.Tinv(mz.max) : (finite(max) ? max : dmax);
    return bounds(pmin,pmax);
  };
}

range Automatic=Range(true,true);
range Full=Range();

void image(frame f, real[][] data, pair initial, pair final, pen[] palette,
           bool transpose=(initial.x < final.x && initial.y < final.y),
           transform t=identity(), bool copy=true, bool antialias=false)
{
  transform T=transpose ? swap : identity();
  _image(f,copy ? copy(data) : data,T*initial,T*final,palette,t*T,copy=false,
         antialias=antialias);
}

void image(frame f, pen[][] data, pair initial, pair final,
           bool transpose=(initial.x < final.x && initial.y < final.y),
           transform t=identity(), bool copy=true, bool antialias=false)
{
  transform T=transpose ? swap : identity();
  _image(f,copy ? copy(data) : data,T*initial,T*final,t*T,copy=false,
         antialias=antialias);
}

// Reduce color palette to approximate range of data relative to "display"
// range => errors of 1/palette.length in resulting color space.
pen[] adjust(picture pic, real min, real max, real rmin, real rmax,
             pen[] palette) 
{
  real dmin=pic.scale.z.T(min);
  real dmax=pic.scale.z.T(max);
  real delta=rmax-rmin;
  if(delta > 0) {
    real factor=palette.length/delta;
    int minindex=floor(factor*(dmin-rmin));
    if(minindex < 0) minindex=0;
    int maxindex=ceil(factor*(dmax-rmin));
    if(maxindex > palette.length) maxindex=palette.length;
    if(minindex > 0 || maxindex < palette.length)
      return palette[minindex:maxindex];
  }
  return palette;
}

private real[] sequencereal;

bounds image(picture pic=currentpicture, real[][] f, range range=Full,
             pair initial, pair final, pen[] palette,
             bool transpose=(initial.x < final.x && initial.y < final.y),
             bool copy=true, bool antialias=false)
{
  if(copy) f=copy(f);
  if(copy) palette=copy(palette);

  real m=min(f);
  real M=max(f);
  bounds bounds=range(pic,m,M);
  real rmin=pic.scale.z.T(bounds.min);
  real rmax=pic.scale.z.T(bounds.max);
  palette=adjust(pic,m,M,rmin,rmax,palette);

  // Crop data to allowed range and scale
  if(range != Full || pic.scale.z.scale.T != identity ||
     pic.scale.z.postscale.T != identity) {
    scalefcn T=pic.scale.z.T;
    real m=bounds.min;
    real M=bounds.max;
    for(int i=0; i < f.length; ++i)
      f[i]=map(new real(real x) {return T(min(max(x,m),M));},f[i]);
  }

  initial=Scale(pic,initial);
  final=Scale(pic,final);

  pic.addBox(initial,final);

  transform T;
  if(transpose) {
    T=swap;
    initial=T*initial;
    final=T*final;
  }
        
  pic.add(new void(frame F, transform t) {
      _image(F,f,initial,final,palette,t*T,copy=false,antialias=antialias);
    },true);
  return bounds; // Return bounds used for color space
}

bounds image(picture pic=currentpicture, real f(real, real),
             range range=Full, pair initial, pair final,
             int nx=ngraph, int ny=nx, pen[] palette, bool antialias=false)
{
  // Generate data, taking scaling into account
  real xmin=pic.scale.x.T(initial.x);
  real xmax=pic.scale.x.T(final.x);
  real ymin=pic.scale.y.T(initial.y);
  real ymax=pic.scale.y.T(final.y);
  real[][] data=new real[ny][nx];
  for(int j=0; j < ny; ++j) {
    real y=pic.scale.y.Tinv(interp(ymin,ymax,(j+0.5)/ny));
    scalefcn Tinv=pic.scale.x.Tinv;
    // Take center point of each bin
    data[j]=sequence(new real(int i) {
        return f(Tinv(interp(xmin,xmax,(i+0.5)/nx)),y);
      },nx);
  }
  return image(pic,data,range,initial,final,palette,transpose=false,
               copy=false,antialias=antialias);
}

void image(picture pic=currentpicture, pen[][] data, pair initial, pair final,
           bool transpose=(initial.x < final.x && initial.y < final.y),
           bool copy=true, bool antialias=false)
{
  if(copy) data=copy(data);

  initial=Scale(pic,initial);
  final=Scale(pic,final);

  pic.addBox(initial,final);

  transform T;
  if(transpose) {
    T=swap;
    initial=T*initial;
    final=T*final;
  }
        
  pic.add(new void(frame F, transform t) {
      _image(F,data,initial,final,t*T,copy=false,antialias=antialias);
    },true);
}

void image(picture pic=currentpicture, pen f(int, int), int width, int height,
           pair initial, pair final,
           bool transpose=(initial.x < final.x && initial.y < final.y),
           bool antialias=false)
{
  initial=Scale(pic,initial);
  final=Scale(pic,final);

  pic.addBox(initial,final);

  transform T;
  if(transpose) {
    T=swap;
    int temp=width;
    width=height;
    height=temp;
    initial=T*initial;
    final=T*final;
  }
        
  pic.add(new void(frame F, transform t) {
      _image(F,f,width,height,initial,final,t*T,antialias=antialias);
    },true);
}

bounds image(picture pic=currentpicture, pair[] z, real[] f,
             range range=Full, pen[] palette)
{
  if(z.length != f.length)
    abort("z and f arrays have different lengths");

  real m=min(f);
  real M=max(f);
  bounds bounds=range(pic,m,M);
  real rmin=pic.scale.z.T(bounds.min);
  real rmax=pic.scale.z.T(bounds.max);

  palette=adjust(pic,m,M,rmin,rmax,palette);
  rmin=max(rmin,m);
  rmax=min(rmax,M);

  // Crop data to allowed range and scale
  if(range != Full || pic.scale.z.scale.T != identity ||
     pic.scale.z.postscale.T != identity) {
    scalefcn T=pic.scale.z.T;
    real m=bounds.min;
    real M=bounds.max;
    f=map(new real(real x) {return T(min(max(x,m),M));},f);
  }

  int[] edges={0,0,1};
  int N=palette.length-1;

  int[][] trn=triangulate(z);
  real step=rmax == rmin ? 0.0 : N/(rmax-rmin);
  for(int i=0; i < trn.length; ++i) {
    int[] trni=trn[i];
    int i0=trni[0], i1=trni[1], i2=trni[2];
    pen color(int i) {return palette[round((f[i]-rmin)*step)];}
    gouraudshade(pic,z[i0]--z[i1]--z[i2]--cycle,
                 new pen[] {color(i0),color(i1),color(i2)},edges);
  }
  return bounds; // Return bounds used for color space
}

bounds image(picture pic=currentpicture, real[] x, real[] y, real[] f,
             range range=Full, pen[] palette)
{
  int n=x.length;
  if(n != y.length)
    abort("x and y arrays have different lengths");

  pair[] z=sequence(new pair(int i) {return (x[i],y[i]);},n);
  return image(pic,z,f,range,palette);
}

// Construct a pen[] array from f using the specified palette.
pen[] palette(real[] f, pen[] palette)
{
  real Min=min(f);
  real Max=max(f);
  if(palette.length == 0) return new pen[];
  real step=Max == Min ? 0.0 : (palette.length-1)/(Max-Min);
  return sequence(new pen(int i) {return palette[round((f[i]-Min)*step)];},
                  f.length);
}

// Construct a pen[][] array from f using the specified palette.
pen[][] palette(real[][] f, pen[] palette)
{
  real Min=min(f);
  real Max=max(f);
  int n=f.length;
  pen[][] p=new pen[n][];
  real step=(Max == Min) ? 0.0 : (palette.length-1)/(Max-Min);
  for(int i=0; i < n; ++i) {
    real[] fi=f[i];
    p[i]=sequence(new pen(int j) {return palette[round((fi[j]-Min)*step)];},
                  f[i].length);
  }
  return p;
}

typedef ticks paletteticks(int sign=-1);

paletteticks PaletteTicks(Label format="", ticklabel ticklabel=null,
                          bool beginlabel=true, bool endlabel=true,
                          int N=0, int n=0, real Step=0, real step=0,
                          pen pTick=nullpen, pen ptick=nullpen)
{
  return new ticks(int sign=-1) {
    format.align(sign > 0 ? RightSide : LeftSide);
    return Ticks(sign,format,ticklabel,beginlabel,endlabel,N,n,Step,step,
                 true,true,extend=true,pTick,ptick);
  };
} 

paletteticks PaletteTicks=PaletteTicks();
paletteticks NoTicks=new ticks(int sign=-1) {return NoTicks;};

void palette(picture pic=currentpicture, Label L="", bounds bounds, 
             pair initial, pair final, axis axis=Right, pen[] palette, 
             pen p=currentpen, paletteticks ticks=PaletteTicks,
             bool copy=true, bool antialias=false)
{
  real initialz=pic.scale.z.T(bounds.min);
  real finalz=pic.scale.z.T(bounds.max);
  bounds mz=autoscale(initialz,finalz,pic.scale.z.scale);
  
  axisT axis;
  axis(pic,axis);
  real angle=degrees(axis.align.dir);

  initial=Scale(pic,initial);
  final=Scale(pic,final);

  pair lambda=final-initial;
  bool vertical=(floor((angle+45)/90) % 2 == 0);
  pair perp,par;

  if(vertical) {perp=E; par=N;} else {perp=N; par=E;}

  path g=(final-dot(lambda,par)*par)--final;
  path g2=initial--final-dot(lambda,perp)*perp;

  if(sgn(dot(lambda,perp)*dot(axis.align.dir,perp)) == -1) {
    path tmp=g;
    g=g2;
    g2=tmp;
  }

  if(copy) palette=copy(palette);
  Label L=L.copy();
  if(L.defaultposition) L.position(0.5);
  L.align(axis.align);
  L.p(p);
  if(vertical && L.defaulttransform) {
    frame f;
    add(f,Label(L.s,(0,0),L.p));
    if(length(max(f)-min(f)) > ylabelwidth*fontsize(L.p)) 
      L.transform(rotate(90));
  }
  real[][] pdata={sequence(palette.length)};
  
  transform T;
  pair Tinitial,Tfinal;
  if(vertical) {
    T=swap;
    Tinitial=T*initial;
    Tfinal=T*final;
  } else {
    Tinitial=initial;
    Tfinal=final;
  }
        
  pic.add(new void(frame f, transform t) {
      _image(f,pdata,Tinitial,Tfinal,palette,t*T,copy=false,
             antialias=antialias);
    },true);
  
  ticklocate locate=ticklocate(initialz,finalz,pic.scale.z,mz.min,mz.max);
  axis(pic,L,g,g2,p,ticks(sgn(axis.side.x*dot(lambda,par))),locate,mz.divisor,
       true);

  pic.add(new void(frame f, transform t) {
      pair Z0=t*initial;
      pair Z1=t*final;
      draw(f,Z0--(Z0.x,Z1.y)--Z1--(Z1.x,Z0.y)--cycle,p);
    },true);

  pic.addBox(initial,final);
}

// A grayscale palette
pen[] Grayscale(int NColors=256)
{
  real ninv=1.0/(NColors-1.0);
  return sequence(new pen(int i) {return gray(i*ninv);},NColors);
}

// A color wheel palette
pen[] Wheel(int NColors=32766)
{
  if(settings.gray) return Grayscale(NColors);
  
  int nintervals=6;
  if(NColors <= nintervals) NColors=nintervals+1;
  int n=-quotient(NColors,-nintervals);
                
  pen[] Palette;
  
  Palette=new pen[n*nintervals];
  real ninv=1.0/n;

  for(int i=0; i < n; ++i) {
    real ininv=i*ninv;
    real ininv1=1.0-ininv;
    Palette[i]=rgb(1.0,0.0,ininv);
    Palette[n+i]=rgb(ininv1,0.0,1.0);
    Palette[2n+i]=rgb(0.0,ininv,1.0);
    Palette[3n+i]=rgb(0.0,1.0,ininv1);
    Palette[4n+i]=rgb(ininv,1.0,0.0);    
    Palette[5n+i]=rgb(1.0,ininv1,0.0);
  }
  return Palette;
}

// A rainbow palette
pen[] Rainbow(int NColors=32766)
{
  if(settings.gray) return Grayscale(NColors);
  
  int offset=1;
  int nintervals=5;
  if(NColors <= nintervals) NColors=nintervals+1;
  int n=-quotient(NColors-1,-nintervals);
                
  pen[] Palette;
  
  Palette=new pen[n*nintervals+offset];
  real ninv=1.0/n;

  for(int i=0; i < n; ++i) {
    real ininv=i*ninv;
    real ininv1=1.0-ininv;
    Palette[i]=rgb(ininv1,0.0,1.0);
    Palette[n+i]=rgb(0.0,ininv,1.0);
    Palette[2n+i]=rgb(0.0,1.0,ininv1);
    Palette[3n+i]=rgb(ininv,1.0,0.0);    
    Palette[4n+i]=rgb(1.0,ininv1,0.0);
  }
  Palette[4n+n]=rgb(1.0,0.0,0.0);
  
  return Palette;
}

private pen[] BWRainbow(int NColors, bool two)
{
  if(settings.gray) return Grayscale(NColors);
  
  int offset=1;
  int nintervals=6;
  int divisor=3;
  
  if(two) nintervals += 6;
  
  int Nintervals=nintervals*divisor;
  if(NColors <= Nintervals) NColors=Nintervals+1;
  int num=NColors-offset;
  int n=-quotient(num,-Nintervals)*divisor;
  NColors=n*nintervals+offset;
                
  pen[] Palette;
  
  Palette=new pen[NColors];
  real ninv=1.0/n;

  int k=0;
  
  if(two) {
    for(int i=0; i < n; ++i) {
      real ininv=i*ninv;
      real ininv1=1.0-ininv;
      Palette[i]=rgb(ininv1,0.0,1.0);
      Palette[n+i]=rgb(0.0,ininv,1.0);
      Palette[2n+i]=rgb(0.0,1.0,ininv1);
      Palette[3n+i]=rgb(ininv,1.0,0.0);
      Palette[4n+i]=rgb(1.0,ininv1,0.0);
      Palette[5n+i]=rgb(1.0,0.0,ininv);
    }
    k += 6n;
  }
  
  if(two)
    for(int i=0; i < n; ++i) 
      Palette[k+i]=rgb(1.0-i*ninv,0.0,1.0);
  else {
    int n3=-quotient(n,-3);
    int n23=2*n3;
    real third=n3*ninv;
    real twothirds=n23*ninv;
    for(int i=0; i < n3; ++i) {
      real ininv=i*ninv;
      Palette[k+i]=rgb(ininv,0.0,ininv);
      Palette[k+n3+i]=rgb(third,0.0,third+ininv);
      Palette[k+n23+i]=rgb(third-ininv,0.0,twothirds+ininv);
    }
  }
  k += n;

  for(int i=0; i < n; ++i) {
    real ininv=i*ninv;
    real ininv1=1.0-ininv;
    Palette[k+i]=rgb(0.0,ininv,1.0);
    Palette[k+n+i]=rgb(0.0,1.0,ininv1);
    Palette[k+2n+i]=rgb(ininv,1.0,0.0);    
    Palette[k+3n+i]=rgb(1.0,ininv1,0.0);
    Palette[k+4n+i]=rgb(1.0,ininv,ininv);
  }
  Palette[k+5n]=rgb(1.0,1.0,1.0);
  
  return Palette;
}

// Quantize palette to exactly n values
pen[] quantize(pen[] Palette, int n)
{
  if(Palette.length == 0) abort("cannot quantize empty palette");
  if(n <= 1) abort("palette must contain at least two pens");
  real step=(Palette.length-1)/(n-1);
  return sequence(new pen(int i) {
      return Palette[round(i*step)];
    },n); 
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

//A palette varying linearly over the specified array of pens, using
// NColors in each interpolation interval.
pen[] Gradient(int NColors=256 ... pen[] p) 
{
  pen[] P;
  if(p.length < 2) abort("at least 2 colors must be specified");
  real step=NColors > 1 ? (1/(NColors-1)) : 1;
  for(int i=0; i < p.length-1; ++i) {
    pen begin=p[i];
    pen end=p[i+1];
    P.append(sequence(new pen(int j) {
          return interp(begin,end,j*step);
        },NColors));
  }
  return P;
}

pen[] cmyk(pen[] Palette) 
{
  int n=Palette.length;
  for(int i=0; i < n; ++i)
    Palette[i]=cmyk(Palette[i]);
  return Palette;
}

import graph3;
import palette; 
import animate;

settings.tex="pdflatex";
settings.render=0;
settings.prc=false;
unitsize(1cm);

animation a;

currentprojection=perspective(-20,-18,18);
currentlight=light(1,1,10);

int n=26;
real L=2.5;
real dx=2*L/n;
real CFL=0.125;
real dt=CFL*dx^2;

real[][] Z=new real[n][n];
real[][] W=new real[n][n];

guide initcond1=shift((-1,-1))*scale(0.5)*unitcircle;
guide initcond2=shift((0.5,0))*yscale(1.2)*unitcircle;

real f(pair p) {return (inside(initcond1,p)||inside(initcond2,p)) ? 2 : 0;}

//Initialize
for(int i=0; i < n; ++i)
  for (int j=0; j < n; ++j)
    Z[i][j]=f((-L,-L)+(2*L/n)*(i,j));

real f(pair t) {
  int i=round((n/2)*(t.x/L+1));
  int j=round((n/2)*(t.y/L+1));
  if(i > n-1) i=n-1;
  if(j > n-1) j=n-1;
  return Z[i][j];
}

surface sf;

void advanceZ(int iter=20) {
  for(int k=0; k < iter; ++k) {
    for(int i=0; i < n; ++i)
      for(int j=0; j < n; ++j)  W[i][j]=0;
    for(int i=1; i < n-1; ++i)
      for(int j=1; j< n-1; ++j) 
        W[i][j]=Z[i][j]+(dt/dx^2)*(Z[i+1][j]+Z[i-1][j]+Z[i][j-1]+Z[i][j+1]
                                   -4*Z[i][j]);
    for(int i=0; i < n; ++i)
      for(int j=0; j < n; ++j)
        Z[i][j]=W[i][j];
  };
}

pen[] Pal=Rainbow(96);

int endframe=40;

for(int fr=0; fr < endframe; ++fr) { 
  if(fr == 0) {// smoothing of initial state; no Spline, but full grid
    advanceZ(3);
    sf=surface(f,(-L,-L),(L,L),nx=n);
  } else // use Spline and fewer grid points to save memory
    sf=surface(f,(-L,-L),(L,L),nx=round(n/2),Spline);
  sf.colors(palette(sf.map(zpart),Pal[0:round(48*max(sf).z)])); 
  draw(sf);
  a.add();
  erase();
  advanceZ(30);
};

label(a.pdf(delay=400,"controls,loop"));
shipout(bbox(3mm,darkblue+3bp+miterjoin,FillDraw(fillpen=paleblue)),"pdf");

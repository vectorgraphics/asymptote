import graph3; 
import animation; 
import solids;

settings.tex="pdflatex"; 
settings.prc=false; 
settings.render=4; 
 
currentprojection=orthographic((0,5,2)); 
currentlight=(0,5,5); 
unitsize(1cm); 
 
real R=3; 
real a=1; 
int n=8; 

path3[] p=new path3[n]; 
animation A; 
 
for(int i=0; i < n; ++i) { 
  triple g(real s) { 
    return ((R-a*cos(2*pi*s))*cos(2*pi/(1+i+s)), 
            (R-a*cos(2*pi*s))*sin(2*pi/(1+i+s)), 
            -a*sin(2*pi*s)); 
  } 
  p[i]=graph(g,0,1,operator ..); 
}

revolution torus=revolution(shift(R*X)*Circle(O,a,Y,32),Z);
surface s=surface(torus);
 
for(int i=0; i < n; ++i){ 
  picture fig; 
  size3(fig,400); 
  draw(fig,s,yellow); 
  for(int j=0;j <= i; ++j)
    draw(fig,p[j],blue+linewidth(4)); 
  A.add(fig); 
}

A.movie(BBox(10,Fill(rgb(0.98,0.98,0.9))),delay=100);

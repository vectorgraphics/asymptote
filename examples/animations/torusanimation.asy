import graph3; 
import animation; 
settings.tex="pdflatex"; 
settings.prc=false; 
settings.render=4; 
 
currentprojection=orthographic((0,5,2)); 
currentlight=(0,5,5); 
unitsize(1cm); 
 
real R=3; 
real a=1; 
triple f(pair t) {
  return ((R+a*cos(t.y))*cos(t.x),(R+a*cos(t.y))*sin(t.x),a*sin(t.y)); 
} 

int n=4; 
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

surface s=surface(f,(0,0),(2pi,2pi),30,20); 
 
for(int i=0; i < n; ++i){ 
  picture fig; 
  size3(fig,400); 
  draw(fig,s,green); 
  draw(fig,p[i],red+linewidth(2)); 
  A.add(fig); 
} 

A.movie(BBox(10,Fill(rgb(0.98,0.98,0.9))),delay=100);

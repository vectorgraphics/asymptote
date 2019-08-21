import graph3; 
import solids; 
import three; 
import animate; 
 
settings.render=2; 
settings.tex="pdflatex"; 
settings.prc=false;
settings.thick=false;
settings.outformat="mpg";
currentprojection=orthographic(5,4,2); 
currentlight=light(specular=black,(0.1,-0.1,1));
 
size(15cm,0); 
 
animation A;
 
real Rst=20, Rl=0.7, Rtl=5; 
real ast=20, est=0.3, bst=ast*sqrt(1-est^2), cst=ast*est; 
real atl=5, etl=0.8, btl=atl*sqrt(1-etl^2), ctl=atl*etl; 
 
real xST(real t) {return ast*cos(t)+cst;} 
real yST(real t) {return bst*sin(t);} 
real zST(real t) {return 0;} 
 
real xTL(real t) {return atl*cos(27t);} 
real yTL(real t) {return btl*sin(27t);} 
real zTL(real t) {return 0;} 
 
 
real xLl(real t) {return Rl*cos(27t);} 
real yLl(real t) {return Rl*sin(27t);} 
real zLl(real t) {return 0;} 
 
real xTt(real t) {return Rtl*cos(100t)/5;} 
real yTt(real t) {return Rtl*sin(100t)/5;} 
real zTt(real t) {return 0;}
 
real xl(real t) {return xST(t)+xTL(t)+xLl(t);} 
real yl(real t) {return yST(t)+yTL(t)+yLl(t);} 
real zl(real t) {return 0;}
 
real xt(real t) {return xST(t)+xTt(t);} 
real yt(real t) {return yST(t)+yTt(t);} 
real zt(real t) {return 0;} 
 
real xL(real t) {return xST(t)+xTL(t);} 
real yL(real t) {return yST(t)+yTL(t);} 
real zL(real t) {return 0;} 
 
path3 Pl=graph(xl,yl,zl,0,2pi,1000),Pt=graph(xt,yt,zt,0,2pi,3000), 
Pts=graph(xST,yST,zST,0,2pi,500); 
 
picture pic;
 
draw(pic,Pl,lightgray); 
draw(pic,Pt,lightblue); 
draw(pic,Pts,blue+dashed); 
 
draw(pic,shift(cst,0,0)*scale3(Rtl/2)*unitsphere,yellow);
 
surface terre=scale3(Rtl/5)*unitsphere; 
surface lune=scale3(Rl)*unitsphere; 
 
int n=50;

real step=2pi/n; 
for(int i=0; i < n; ++i) { 
  real k=i*step; 
  add(pic); 
  draw(shift(xL(k),yL(k),0)*lune,lightgray); 
  draw(shift(xST(k),yST(k),0)*terre,lightblue+lightgreen); 
  A.add(); 
  erase(); 
} 

A.movie(BBox(1mm,Fill(Black)),delay=500,
        options="-density 288x288 -geometry 50%x");

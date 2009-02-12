import graph; 
import palette;
texpreamble("\usepackage[amssymb,thinqspace,thinspace]{SIunits}"); 
 
size(800,200); 
 
real c=3e8;
real nm=1e-9;
real freq(real lambda) {return c/(lambda*nm);} 
real lambda(real f) {return c/(f*nm);} 
 
real fmin=10; 
real fmax=1e23; 
 
scale(Log(true),Linear(true)); 
xlimits(fmin,fmax); 
ylimits(0,1); 
 
real uv=freq(400);
real ir=freq(700);
 
bounds visible=bounds(Scale(uv).x,Scale(ir).x);
palette(visible,uv,ir+(0,2),Bottom,Rainbow(),invisible);

xaxis(Label("\hertz",1),Bottom,RightTicks,above=true); 
 
real log10Left(real x) {return -log10(x);}
real pow10Left(real x) {return pow10(-x);}

scaleT LogLeft=scaleT(log10Left,pow10Left,logarithmic=true);

picture q=secondaryX(new void(picture p) { 
    scale(p,LogLeft,Linear); 
    xlimits(p,lambda(fmax),lambda(fmin));
    ylimits(p,0,1); 
    xaxis(p,Label("\nano\metre",1),Top,LeftTicks(DefaultLogFormat,n=10)); 
  }); 
 
add(q,above=true); 

margin margin=PenMargin(0,0);
draw("radio",Scale((10,1))--Scale((5e12,1)),N,Arrow); 
draw("infrared",Scale((1e12,1.5))--Scale(shift(0,1.5)*ir),Arrows,margin);
draw("UV",Scale(shift(0,1.5)*uv)--Scale((1e17,1.5)),Arrows,margin);
draw("x-rays",Scale((1e16,1))--Scale((1e21,1)),N,Arrows); 
draw("$\gamma$-rays",Scale((fmax,1.5))--Scale((2e18,1.5)),Arrow); 


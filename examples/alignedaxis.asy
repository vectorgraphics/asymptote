import graph;

real Freq=60.0;
real margin=5mm;

pair exp(pair x) {
  return exp(x.x)*(cos(x.y)+I*sin(x.y));
}

real Merr(real x, real w) {
  real tau=x/(2*Freq);
  return 20*log(abs((tau*w+tau/(exp(I*2*pi*Freq*tau)-1))*(I*2*pi*Freq)));
}

real Aerr(real x, real w) {
  real tau=x/(2*Freq);
  return degrees((tau*w+tau/(exp(I*2*pi*Freq*tau)-1))*(I*2*pi*Freq));
}

picture pic1;
scale(pic1,Log,Linear);
real Merr1(real x){return Merr(x,1);}
draw(pic1,graph(pic1,Merr1,1e-4,1),black+1.2);

ylimits(pic1,-60,20);
yaxis(pic1,"magnitude (dB)",LeftRight,RightTicks(new
                                                 real[] {-60,-40,-20,0,20}));
xaxis(pic1,"$f/f_\mathrm{Ny}$",BottomTop,LeftTicks(N=5));
yequals(pic1,0,Dotted);
yequals(pic1,-20,Dotted);
yequals(pic1,-40,Dotted);
xequals(pic1,1e-3,Dotted);
xequals(pic1,1e-2,Dotted);
xequals(pic1,1e-1,Dotted);

size(pic1,100,100,point(pic1,SW),point(pic1,NE));

label(pic1,"$\theta=1$",point(pic1,N),2N);

frame f1=pic1.fit();
add(f1);

picture pic1p;
scale(pic1p,Log,Linear);
real Aerr1(real x){return Aerr(x,1);}
draw(pic1p,graph(pic1p,Aerr1,1e-4,1),black+1.2);

ylimits(pic1p,-5,95);
yaxis(pic1p,"phase (deg)",LeftRight,RightTicks(new real[] {0,45,90}));
xaxis(pic1p,"$f/f_\mathrm{Ny}$",BottomTop,LeftTicks(N=5));
yequals(pic1p,0,Dotted);
yequals(pic1p,45,Dotted);
yequals(pic1p,90,Dotted);
xequals(pic1p,1e-3,Dotted);
xequals(pic1p,1e-2,Dotted);
xequals(pic1p,1e-1,Dotted);

size(pic1p,100,100,point(pic1p,SW),point(pic1p,NE));

frame f1p=pic1p.fit();
f1p=shift(0,min(f1).y-max(f1p).y-margin)*f1p;
add(f1p);

picture pic2;
scale(pic2,Log,Linear);
real Merr2(real x){return Merr(x,0.75);}
draw(pic2,graph(pic2,Merr2,1e-4,1),black+1.2);

ylimits(pic2,-60,20);
yaxis(pic2,"magnitude (dB)",LeftRight,RightTicks(new
                                                 real[] {-60,-40,-20,0,20}));
xaxis(pic2,"$f/f_\mathrm{Ny}$",BottomTop,LeftTicks(N=5));
yequals(pic2,0,Dotted);
yequals(pic2,-20,Dotted);
yequals(pic2,-40,Dotted);
xequals(pic2,1e-3,Dotted);
xequals(pic2,1e-2,Dotted);
xequals(pic2,1e-1,Dotted);

size(pic2,100,100,point(pic2,SW),point(pic2,NE));

label(pic2,"$\theta=0.75$",point(pic2,N),2N);

frame f2=pic2.fit();
f2=shift(max(f1).x-min(f2).x+margin)*f2;
add(f2);

picture pic2p;
scale(pic2p,Log,Linear);
real Aerr2(real x){return Aerr(x,0.75);}
draw(pic2p,graph(pic2p,Aerr2,1e-4,1),black+1.2);

ylimits(pic2p,-5,95);
yaxis(pic2p,"phase (deg)",LeftRight,RightTicks(new real[] {0,45.1,90}));
xaxis(pic2p,"$f/f_\mathrm{Ny}$",BottomTop,LeftTicks(N=5));
yequals(pic2p,0,Dotted);
yequals(pic2p,45,Dotted);
yequals(pic2p,90,Dotted);
xequals(pic2p,1e-3,Dotted);
xequals(pic2p,1e-2,Dotted);
xequals(pic2p,1e-1,Dotted);

size(pic2p,100,100,point(pic2p,SW),point(pic2p,NE));

frame f2p=pic2p.fit();
f2p=shift(max(f1p).x-min(f2p).x+margin,min(f2).y-max(f2p).y-margin)*f2p;
add(f2p);

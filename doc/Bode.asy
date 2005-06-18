import graph;
texpreamble("\def\Arg{\mathop {\rm Arg}\nolimits}");

size(10cm,5cm,IgnoreAspect);

real ampl(real x) {return 1/(1+x^2);}
real phas(real x) {return -atan(x)/pi;}

scale(Log,Log);
draw(graph(ampl,0.01,10));
ylimits(.001,100);

xaxis("$\omega\tau_0$",BottomTop,LeftTicks);
yaxis("$|G(\omega\tau_0)|$",Left,RightTicks);

picture q=secondaryY(new void(picture p) {
		       scale(p,Log,Linear);
		       draw(p,graph(p,phas,0.01,10),red);
		       ylimits(p,-1,1.5);
		       yaxis(p,"$\Arg G/\pi$",black,red,Right,
			     LeftTicks("% #.1f"));
		       xaxis(p,Dotted,YEquals(1,false));
		     });
label(q,"(1,0)",Scale(q,(1,0)),red);
add(q);

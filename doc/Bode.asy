import graph;
texpreamble("\def\Arg{\mathop {\rm Arg}\nolimits}");

size(10cm,5cm,IgnoreAspect);

real ampl(real x) {return 2.5/(1+x^2);}
real phas(real x) {return -atan(x)/pi;}

scale(Log,Log);
draw(graph(ampl,0.01,10));
ylimits(.001,100);

xaxis("$\omega\tau_0$",BottomTop,LeftTicks);
yaxis("$|G(\omega\tau_0)|$",Left,RightTicks);

picture q=secondaryY(new void(picture pic) {
		       scale(pic,Log,Linear);
		       draw(pic,graph(pic,phas,0.01,10),red);
		       ylimits(pic,-1.0,1.5);
		       yaxis(pic,"$\Arg G/\pi$",red,Right,
			     LeftTicks(false,false,"$% #.1f$"));
		       yequals(pic,1,Dotted);
		     });
label(q,"(1,0)",Scale(q,(1,0)),red);
add(q);

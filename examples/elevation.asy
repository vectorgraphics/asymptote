import graph3;
import grid3;
import palette;

currentprojection=orthographic(0.8,1,1);

size(400,300,IgnoreAspect);

real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);}

surface s=surface(f,(-1/2,-1/2),(1/2,1/2),50,Spline);

draw(s,palette(sequence(new real(int i) {return s.s[i].cornermean().z;},
			s.s.length),Rainbow()),black);

grid3(XYZgrid);

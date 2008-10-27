import graph3;
import palette;
size(200);

currentprojection=orthographic(4,2,4);

triple f(pair z) {return expi(z.x,z.y);}

surface s=surface(f,(0,0),(pi,2pi),30);
draw(s,palette(sequence(new real(int i) {return s.s[i].cornermean().z;},
			s.s.length),BWRainbow()),black,nolight);

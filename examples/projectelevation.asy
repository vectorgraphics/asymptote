import graph3; 
import grid3; 
import palette; 
 
currentprojection=orthographic(0.8,1,2); 
 
size(400,300,IgnoreAspect); 
 
real f(pair z) {return cos(2*pi*z.x)*sin(2*pi*z.y);} 
 
surface s=surface(f,(-1/2,-1/2),(1/2,1/2),50,Spline); 
draw(s,lightgray+opacity(0.7)); 

draw(planeproject(unitsquare3)*s,
     palette(sequence(new real(int i) {return s.s[i].cornermean().z;},
		      s.s.length),Rainbow()),nolight);
grid3(XYZgrid); 
 

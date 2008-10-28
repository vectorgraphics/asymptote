import solids; 
import palette; 
 
currentprojection=orthographic(20,0,3); 
 
size(400,300,IgnoreAspect); 
 
revolution r=revolution(new real(real x) {return sin(x)*exp(-x/2);},
			  0,2pi,operator ..,Z); 
surface s=surface(r); 
 
draw(s,lightgray); 
draw(planeproject(shift(-Z)*unitsquare3)*s,
     palette(sequence(new real(int i) {return s.s[i].cornermean().z;}, 
		      s.s.length),Rainbow()));

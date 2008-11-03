import solids; 
import palette; 
 
currentprojection=orthographic(20,0,3); 
 
size(400,300,IgnoreAspect); 
 
revolution r=revolution(new real(real x) {return sin(x)*exp(-x/2);},
			  0,2pi,operator ..,Z); 
surface s=surface(r); 
 
surface S=planeproject(shift(-Z)*unitsquare3)*s;
S.colors(palette(s.map(zpart),Rainbow()));
draw(S);
draw(s,lightgray); 

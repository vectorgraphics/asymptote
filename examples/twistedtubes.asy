import graph3;  
import palette; 
 
size(300,300,keepAspect=true);  
 
real w=0.4;

real f(triple t) {return sin(t.x);} 
triple f1(pair t) {return (cos(t.x)-2cos(w*t.y),sin(t.x)-2sin(w*t.y),t.y);}
triple f2(pair t) {return (cos(t.x)+2cos(w*t.y),sin(t.x)+2sin(w*t.y),t.y);}
triple f3(pair t) {return (cos(t.x)+2sin(w*t.y),sin(t.x)-2cos(w*t.y),t.y);}
triple f4(pair t) {return (cos(t.x)-2sin(w*t.y),sin(t.x)+2cos(w*t.y),t.y);}

surface s1=surface(f1,(0,0),(2pi,10),25,15); 
surface s2=surface(f2,(0,0),(2pi,10),25,15); 
surface s3=surface(f3,(0,0),(2pi,10),25,15); 
surface s4=surface(f4,(0,0),(2pi,10),25,15); 

pen[] Rainbow=Rainbow();
s1.colors(palette(s1.map(f),Rainbow)); 
s2.colors(palette(s2.map(f),Rainbow)); 
s3.colors(palette(s3.map(f),Rainbow)); 
s4.colors(palette(s4.map(f),Rainbow)); 

draw(s1,meshpen=black); 
draw(s2,meshpen=black); 
draw(s3,meshpen=black); 
draw(s4,meshpen=black); 

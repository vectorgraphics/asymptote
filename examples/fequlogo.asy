import graph3;
import obj;

size(200,0);
size3(200);

if(settings.render < 0) settings.render=8; 

texpreamble("\usepackage[T1]{fontenc}");
texpreamble("\usepackage{ccfonts,eulervm}");

currentprojection=perspective(4,1,2);
currentlight=(4,0,2);
currentlight.background=black+opacity(0.0);

real R=4;

triple f1(pair t) {return (R*cos(t.x),R*sin(t.x),t.y);}

draw(shift(-0.6Z)*scale3(0.66)*rotate(55,Z)*rotate(90,X)*
     obj("uhrturm.obj",orange));

surface s=surface(f1,(0,0),(2pi,2),8,8,Spline);

string lo="$\displaystyle f(x+y)=f(x)+f(y)$";
string hi="$\displaystyle F_{t+s}=F_t\circ F_s$";

real h=0.0125;

draw(surface(rotate(2)*xscale(0.32)*yscale(0.6)*lo,s,-pi/4-1.5*pi/20,0.5,h));
draw(surface(rotate(0)*xscale(-0.45)*yscale(0.3)*hi,s,0.8*pi,0.25,h),blue);

add(new void(frame f, transform3 t, picture pic, projection P) {
    draw(f,surface(invert(box(min(f,P),max(f,P)),min3(f),P),
                   new pen[] {orange,red,yellow,brown}+opacity(0.9)));
  }
);

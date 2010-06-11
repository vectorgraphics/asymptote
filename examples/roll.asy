import graph3;

size(200,0);

triple f(pair t) {
return(t.x+t.y/4+sin(t.y),cos(t.y),sin(t.y));
}

surface s=surface(f,(0,0),(2pi,2pi),7,20,Spline);
draw(s,olive,render(merge=true));

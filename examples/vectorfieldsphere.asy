import graph3;

size(12cm);

currentprojection=orthographic(1,-2,1);
currentlight=(1,-1,0.5);

triple f(pair z) {return expi(z.x,z.y);}

path3 vector(pair z) {
  triple v=f(z);
  return O--(v.y,v.z,v.x);
}

add(vectorfield(vector,f,(0,0),(pi,2pi),10,0.25,red,render(merge=true)));

draw(unitsphere,gray+opacity(0.5),render(compression=0,merge=true));

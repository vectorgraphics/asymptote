import solids;
import animation;

currentprojection=orthographic((0,5,2));
currentlight=(0,5,5);

settings.thick=false;
settings.render=0;

int nbpts=200;
real step=2*pi/nbpts;
int angle=10;

unitsize(1cm);

triple[] P=new triple[nbpts];
for(int i=0; i < nbpts; ++i) {
  real t=-pi+i*step;
  P[i]=(3sin(t)*cos(2t),3sin(t)*sin(2t),3cos(t));
}

transform3 t=rotate(angle,(0,0,0),(1,0.25,0.25));
revolution r=sphere(O,3);
draw(surface(r),lightgrey);
draw(r,backpen=linetype("8 8",8));

animation A;

for(int phi=0; phi < 360; phi += angle) {
  bool[] front=new bool[nbpts];
  save();
  for(int i=0; i < nbpts; ++i) {
    P[i]=t*P[i];
    front[i]=dot(P[i],currentprojection.camera) > 0;
  }
  draw(segment(P,front,operator ..),1mm+blue+extendcap);
  draw(segment(P,!front,operator ..),grey);
  A.add();
  restore();
}

A.movie(0,200);
currentpicture.erase();

import math;
import bsp;
import animation;

size(100,100);

animation a;

void face(face[] faces, path3 p, int j, int k) {
  picture pic=faces.push(p);
  filldraw(pic,project(p),Pen(j));
  int sign=(k % 2 == 0) ? 1 : -1;
  transform t=scale(4)*transform(dir(p,0,sign),dir(p,0,-sign));
  label(pic,t*(string) j,project(0.5*(min(p)+max(p))));
}

void snapshot(transform3 t)
{
  static transform3 s=shift(-0.5*(X+Y+Z));
  save();
  
  face[] faces;
  int j=-1;
  transform3 T=t*s;
  for(int k=0; k < 2; ++k) {
    face(faces,T*plane((1,0,0),(0,1,0),(0,0,k)),++j,k);
    face(faces,T*plane((0,1,0),(0,0,1),(k,0,0)),++j,k);
    face(faces,T*plane((0,0,1),(1,0,0),(0,k,0)),++j,k); 
  }
  add(faces);
  
  a.add();
  restore();
}

int n=50;

real step=360/n;
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,X));
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,Y));
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,Z));

a.movie(loops=10,delay=50);

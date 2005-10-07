import math;
import three;

void face(face[] faces, path3 p, int j) {
  picture pic=faces.push(p);
  filldraw(pic,p,Pen(j));
}

void snapshot(transform3 t)
{
  static transform3 s=shift(-0.5*(X+Y+Z));
  static int count=-1;

  picture pic;
  size(pic,100,100);
  
  face[] faces;
  int j=-1;
  transform3 T=t*s;
  for(int k=0; k < 2; ++k) {
    face(faces,T*plane((0,0,1),(0,1,0),(k,0,0)),++j);
    face(faces,T*plane((0,0,1),(1,0,0),(0,k,0)),++j); 
    face(faces,T*plane((0,1,0),(1,0,0),(0,0,k)),++j);
  }
  add(pic,faces);
  draw(pic,box((-1,-1),(1,1)),invisible);
  shipout(fileprefix()+(string) ++count,pic,"gif",quiet=true);
}

int n=100;

real step=360/n;
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,X));
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,Y));
for(int i=0; i < n; ++i)
  snapshot(rotate(i*step,Z));

gifmerge(10,5);

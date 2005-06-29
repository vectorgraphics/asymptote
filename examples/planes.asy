size(8cm,0);
import math;
import graph3d;

real u=2.5;
real v=1;

vector[] FaceY=
new vector[] {vector(-u,-v,0),vector(-u,v,0),vector(u,v,0),vector(u,-v,0)};
vector[] FaceG=
new vector[] {vector(0,-u,-v),vector(0,-u,v),vector(0,u,v),vector(0,u,-v)};
vector[] FaceA=
new vector[] {vector(-v,0,-u),vector(-v,0,u),vector(v,0,u),vector(v,0,-u)};

piclist a=new piclist;
picture pic;
int[] Yindex,Gindex,Aindex;

pic;
filldraw(pic,P(FaceY)--cycle,yellow); a.push(pic,Yindex);

pic;
filldraw(pic,P(FaceA)--cycle,lightgrey); a.push(pic,Aindex);

pic;
filldraw(pic,P(FaceG)--cycle,green); a.push(pic,Gindex);

splitplanes(a,FaceY,Yindex,FaceG,Gindex);
splitplanes(a,FaceG,Gindex,FaceA,Aindex);
splitplanes(a,FaceA,Aindex,FaceY,Yindex);


for(int L=a.maxlevel; L >= 0; --L) {
  for(int i=0; i < a.length(); ++i) {
    if(a.list[i].level == L) add(a.list[i].pic);
  }
}


size(8cm,0);
import math;
import three;

// TODO: reimplement hidden surface removal with binary space partition

real u=2.5;
real v=1;

currentprojection=oblique;

triple[] FaceY=new triple[] {(-u,-v,0),(-u,v,0),(u,v,0),(u,-v,0)};
triple[] FaceG=new triple[] {(0,-u,-v),(0,-u,v),(0,u,v),(0,u,-v)};
triple[] FaceA=new triple[] {(-v,0,-u),(-v,0,u),(v,0,u),(v,0,-u)};

//piclist a=new piclist;
picture pic;
int[] Yindex,Gindex,Aindex;

filldraw(pic,operator --(... FaceY)--cycle3,yellow);// a.push(pic,Yindex);

//pic.erase();
filldraw(pic,operator --(... FaceA)--cycle3,lightgrey);// a.push(pic,Aindex);

//pic.erase;
filldraw(pic,operator -- (... FaceG)--cycle3,green);// a.push(pic,Gindex);

/*
splitplanes(a,FaceY,Yindex,FaceG,Gindex);
splitplanes(a,FaceG,Gindex,FaceA,Aindex);
splitplanes(a,FaceA,Aindex,FaceY,Yindex);


for(int L=a.maxlevel; L >= 0; --L) {
  for(int i=0; i < a.length(); ++i) {
    if(a.list[i].level == L) add(a.list[i].pic);
  }
}
*/

add(pic);

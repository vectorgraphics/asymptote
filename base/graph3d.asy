// Three-dimensional graphics routines
import math;

public pair projection=-dir(45)/sqrt(2);

pair P(real x, real y, real z)
{
  return x*(1,0)+y*(0,1)+projection*z;
}

pair P(vector v) 
{
  return P(v.x,v.y,v.z);
}

guide box3d(vector v1, vector v2)
{
  return 
    P(v1.x,v1.y,v1.z)--
    P(v1.x,v1.y,v2.z)--
    P(v1.x,v2.y,v2.z)--
    P(v1.x,v2.y,v1.z)--
    P(v1.x,v1.y,v1.z)--
    P(v2.x,v1.y,v1.z)--
    P(v2.x,v1.y,v2.z)--
    P(v2.x,v2.y,v2.z)--
    P(v2.x,v2.y,v1.z)--
    P(v2.x,v1.y,v1.z)--
    P(v2.x,v2.y,v1.z)--
    P(v1.x,v2.y,v1.z)--
    P(v1.x,v2.y,v2.z)--
    P(v2.x,v2.y,v2.z)--
    P(v2.x,v1.y,v2.z)--
    P(v1.x,v1.y,v2.z);
}

guide P(vector[] p) {
    guide g;
    for(int i=0; i < p.length; ++i) {
      g=g--P(p[i]);
    }
    return g;
}

struct picitem {
  public picture pic;
  public int level=0;
}

struct piclist {
  picitem[] list;
  pair userMin=infinity, userMax=-infinity;
  int maxlevel=0;

  void push(picture pic) {
    picitem p=new picitem;
    p.pic=pic.copy();
    list.push(p);
  }
  
  void push(picture pic, int[] index) {
    userMin=minbound(userMin,pic.userMin);
    userMax=maxbound(userMax,pic.userMax);
    index.push(list.length);
    push(pic);
  }
    
  void push(picture pic, int i) {
    push(pic);
    if(i >= 0) list[-1].level=list[i].level;
  }
  
  void cover(int i) {maxlevel=max(maxlevel,++list[i].level);}
  
  int length() {return list.length; }
}

void splitplanes(piclist Pic, vector[] a, int[] aindex,
		 vector[] b, int[] bindex)
{
  vector na=normal(a);
  vector nb=normal(b);
  vector Z=intersectionpoint(na,a[0],nb,b[0]);
  if(Z.x == infinity) {abort("Parallel plane case not yet implemented");}
  pair z=P(Z);
  vector Dir=cross(na,nb);
  pair lambda=abs(maxbound(Pic.userMax,z)-minbound(Pic.userMin,z))
    *unit(P(Dir));
  
  // Determine clipping half-planes
  vector ta=cross(Dir,na);
  vector tb=cross(Dir,nb);
  pair T=((dot(P(ta),P(tb)) > 0) ^ (ta.z > tb.z) ? -I : I)*lambda;
  guide g1=z-lambda--z+lambda--z+lambda+T--z-lambda+T--cycle;
  guide g2=z-lambda--z+lambda--z+lambda-T--z-lambda-T--cycle;
  
  int n=aindex.length;
  for(int i=0; i < n; ++i) {
    int I=aindex[i];
    picture bottom=Pic.list[I].pic;
    picture top=new picture;
    add(top,bottom);
    clip(bottom,g2);
    clip(top,g1);
    aindex.push(Pic.length());
    Pic.push(top,I);
    Pic.cover(I);
  }
  
  n=bindex.length;
  for(int i=0; i < n; ++i) {
    int I=bindex[i];
    picture bottom=Pic.list[I].pic;
    picture top=new picture;
    add(top,bottom);
    clip(bottom,g1);
    clip(top,g2);
    bindex.push(Pic.length());
    Pic.push(top,I);
    Pic.cover(I);
  }
}

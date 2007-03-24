import three;
import light;
import graph_settings;

struct surface {
  triple[][] P=new triple[4][4];

  void init(triple[][] P) {
    this.P=copy(P);
  }

  void init(path3 external, triple[] internal=new triple[]) {
    if(internal.length == 0) {
      for(int j=0; j < 4; ++j) {
        static real nineth=1.0/9.0;
        internal[j]=nineth*(-4.0*point(external,j)
                            +6.0*(precontrol(external,j)+
                                  postcontrol(external,j))
                            -2.0*(point(external,j-1)+point(external,j+1))
                            +3.0*(precontrol(external,j-1)+
                                  postcontrol(external,j+1))-
                            point(external,j+2));
      }
    }

    P[1][0]=precontrol(external,0);
    P[0][0]=point(external,0);
    P[0][1]=postcontrol(external,0);
    P[1][1]=internal[0];

    P[0][2]=precontrol(external,1);
    P[0][3]=point(external,1);
    P[1][3]=postcontrol(external,1);
    P[1][2]=internal[1];

    P[2][3]=precontrol(external,2);
    P[3][3]=point(external,2);
    P[3][2]=postcontrol(external,2);
    P[2][2]=internal[2];

    P[3][1]=precontrol(external,3);
    P[3][0]=point(external,3);
    P[2][0]=postcontrol(external,3);
    P[2][1]=internal[3];
  }

  path3 external() {
    return
      P[0][0]..controls P[0][1] and P[0][2]..
      P[0][3]..controls P[1][3] and P[2][3]..
      P[3][3]..controls P[3][2] and P[3][1]..
      P[3][0]..controls P[2][0] and P[1][0]..cycle3;
  }

  triple[] internal() {
    return new triple[]{P[1][1],P[1][2],P[2][2],P[2][1]};
  }

  triple Bezier(triple a, triple b,triple c, triple d, real t) {
    return a*(1-t)^3+b*3t*(1-t)^2+c*3t^2*(1-t)+d*t^3;
  }

  path3 uequals(real u) {
    triple Bu(int j, real u) {return Bezier(P[0][j],P[1][j],P[2][j],P[3][j],u);}
    return Bu(0,u)..controls Bu(1,u) and Bu(2,u)..Bu(3,u);
  }

  path3 vequals(real v) {
    triple Bv(int i, real v) {return Bezier(P[i][0],P[i][1],P[i][2],P[i][3],v);}
    return Bv(0,v)..controls Bv(1,v) and Bv(2,v)..Bv(3,v);
  }

  pen[] colors(pen surfacepen=lightgray, light light=currentlight) {
    pen color(triple dfu, triple dfv) {
      return light.intensity(cross(dfu,dfv))*surfacepen;
    }

    return new pen[] {
      color(P[1][0]-P[0][0],P[0][1]-P[0][0]),
        color(P[1][3]-P[0][3],P[0][3]-P[0][2]),
        color(P[3][3]-P[2][3],P[3][3]-P[3][2]),
        color(P[3][0]-P[2][0],P[3][1]-P[3][0])};
  };

  triple min() {return minbound(P);}
  triple max() {return maxbound(P);}
}

surface operator * (transform3 t, surface s)
{ 
  surface S;
  triple[][] p=s.P;
  triple[][] P=S.P;
  for(int i=0; i < p.length; ++i) { 
    triple[] si=p[i];
    triple[] Si=P[i];
    for(int j=0; j < si.length; ++j) { 
      Si[j]=t*si[j]; 
    } 
  }
  return S; 
}
 
surface operator cast(triple[][] P) {
  surface s;
  s.init(P);
  return s;
}

surface surface(triple[][] P) {
  surface s;
  s.init(P);
  return s;
}

surface surface(path3 external, triple[] internal=new triple[]) 
{
  surface s;
  s.init(external,internal);
  return s;
}

triple min(surface s) {return s.min();}
triple max(surface s) {return s.max();}

surface subsurfaceu(surface s, real ua, real ub)
{
  path3 G=s.uequals(ua)&subpath(s.vequals(1),ua,ub)&
    reverse(s.uequals(ub));
  path3 w=subpath(s.vequals(0),ub,ua);
  path3 i1=s.P[0][1]..controls s.P[1][1] and s.P[2][1]..s.P[3][1];
  path3 i2=s.P[0][2]..controls s.P[1][2] and s.P[2][2]..s.P[3][2];
  path3 s1=subpath(i1,ua,ub);
  path3 s2=subpath(i2,ua,ub);
  return surface(G..controls postcontrol(w,0) and precontrol(w,1)..cycle3,
                 new triple[] {postcontrol(s1,0),postcontrol(s2,0),
                     precontrol(s2,1),precontrol(s1,1)});
}

surface subsurfacev(surface s, real va, real vb)
{
  path3 G=subpath(s.uequals(0),va,vb)&s.vequals(vb)&
    subpath(s.uequals(1),vb,va);
  path3 w=s.vequals(va);
  path3 j1=s.P[1][0]..controls s.P[1][1] and s.P[1][2]..s.P[1][3];
  path3 j2=s.P[2][0]..controls s.P[2][1] and s.P[2][2]..s.P[2][3];
  path3 t1=subpath(j1,va,vb);
  path3 t2=subpath(j2,va,vb);

  return surface(G..controls precontrol(w,1) and postcontrol(w,0)..cycle3,
                 new triple[] {postcontrol(t1,0),precontrol(t1,1),
                     precontrol(t2,1),postcontrol(t2,0)});
}

surface subsurface(surface s, real ua, real ub, real va, real vb)
{
  return subsurfaceu(subsurfacev(s,va,vb),ua,ub);
}

triple point(surface s, real u, real v)
{
  return point(s.uequals(u),v);
}

void tensorshade(picture pic=currentpicture, surface s,
                 pen surfacepen=lightgray, light light=currentlight,
                 projection P=currentprojection)
{
  path[] b=project(box(min(s),max(s)),P);
  tensorshade(pic,box(min(b),max(b)),surfacepen,s.colors(surfacepen,light),
              project(s.external(),P),project(s.internal(),P));
}

void draw(picture pic=currentpicture, surface s, int nu=nmesh, int nv=nu,
          pen surfacepen=lightgray, pen meshpen=nullpen,
          light light=currentlight, projection P=currentprojection)
{
  // Draw a mesh in the absence of lighting (override with meshpen=invisible).
  if(light.source == O && meshpen == nullpen) meshpen=currentpen;

  if(surfacepen != nullpen && nu > 0) {
    // Sort cells by mean distance from camera
    triple camera=P.camera;
    if(P.infinity)
      camera *= max(abs(min(s)),abs(max(s)));

    real[][] depth;
    surface[] su=new surface[nu];
    
    for(int i=0; i < nu; ++i) {
      su[i]=subsurfaceu(s,i/nu,(i+1)/nu);
      path3 s0=s.uequals(i/nu);
      path3 s1=s.uequals((i+1)/nu);
      for(int j=0; j < nv; ++j) {
        triple v=camera-0.25*(point(s0,j/nv)+point(s0,(j+1)/nv)+
			      point(s1,j/nv)+point(s1,(j+1)/nv));
        real d=sgn(dot(v,camera))*abs(v);
        depth.push(new real[] {d,i,j});
      }
    }

    depth=sort(depth);

    // Draw from farthest to nearest
    while(depth.length > 0) {
      real[] a=depth.pop();
      int i=round(a[1]);
      int j=round(a[2]);
      tensorshade(pic,subsurfacev(su[i],j/nv,(j+1)/nv),surfacepen,light,P);
    }
  }

  if(meshpen != nullpen) {
    real step=nu == 0 ? 0 : 1/nu;
    for(int i=0; i <= nu; ++i)
      draw(pic,s.uequals(i*step),meshpen);
    
    real step=nv == 0 ? 0 : 1/nv;
    for(int j=0; j <= nv; ++j)
      draw(pic,s.vequals(j*step),meshpen);
  }
}

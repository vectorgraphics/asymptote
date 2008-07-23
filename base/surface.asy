import three;
import light;
import graph_settings;
import bezulate;

int maxdepth=16;

private void abortcyclic() {abort("cyclic path of length 4 expected");}

struct patch {
  triple[][] P=new triple[4][4];

  path3 external() {
    return
      P[0][0]..controls P[0][1] and P[0][2]..
      P[0][3]..controls P[1][3] and P[2][3]..
      P[3][3]..controls P[3][2] and P[3][1]..
      P[3][0]..controls P[2][0] and P[1][0]..cycle;
  }

  triple[] internal() {
    return new triple[]{P[1][1],P[1][2],P[2][2],P[2][1]};
  }

  triple[] controlpoints() {
    return new triple[] {P[0][0],P[0][1],P[0][2],P[0][3],
	P[1][0],P[1][1],P[1][2],P[1][3],
	P[2][0],P[2][1],P[2][2],P[2][3],
	P[3][0],P[3][1],P[3][2],P[3][3]};
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


  private real[] split(real z0, real c0, real c1, real z1) {
    real m0=0.5*(z0+c0);
    real m1=0.5*(c0+c1);
    real m2=0.5*(c1+z1);
    real m3=0.5*(m0+m1);
    real m4=0.5*(m1+m2);
    real m5=0.5*(m3+m4);

    return new real[] {m0,m3,m5,m4,m2};
  }

  triple[] split(triple z0, triple c0, triple c1, triple z1) {
    triple m0=0.5*(z0+c0);
    triple m1=0.5*(c0+c1);
    triple m2=0.5*(c1+z1);
    triple m3=0.5*(m0+m1);
    triple m4=0.5*(m1+m2);
    triple m5=0.5*(m3+m4);

    return new triple[] {m0,m3,m5,m4,m2};
  }

  // Split one component of a Bezier patch into 4 subpatches.
  real[][] splitpatch4(real[] p) {
    // Find new control points.
    real[] c0=split(p[0],p[1],p[2],p[3]);
    real[] c1=split(p[4],p[5],p[6],p[7]);
    real[] c2=split(p[8],p[9],p[10],p[11]);
    real[] c3=split(p[12],p[13],p[14],p[15]);

    real[] c4=split(p[12],p[8],p[4],p[0]);
    real[] c5=split(c3[0],c2[0],c1[0],c0[0]);
    real[] c6=split(c3[1],c2[1],c1[1],c0[1]);
    real[] c7=split(c3[2],c2[2],c1[2],c0[2]);

    real[] c8=split(c3[3],c2[3],c1[3],c0[3]);
    real[] c9=split(c3[4],c2[4],c1[4],c0[4]);
    real[] c10=split(p[15],p[11],p[7],p[3]);

    // Set up 4 Bezier subpatches.
    real[] s0={c4[2],c5[2],c6[2],c7[2],c4[1],c5[1],c6[1],c7[1],
               c4[0],c5[0],c6[0],c7[0],p[12],c3[0],c3[1],c3[2]};
    real[] s1={p[0],c0[0],c0[1],c0[2],c4[4],c5[4],c6[4],c7[4],
               c4[3],c5[3],c6[3],c7[3],c4[2],c5[2],c6[2],c7[2]};
    real[] s2={c0[2],c0[3],c0[4],p[3],c7[4],c8[4],c9[4],c10[4],
               c7[3],c8[3],c9[3],c10[3],c7[2],c8[2],c9[2],c10[2]};
    real[] s3={c7[2],c8[2],c9[2],c10[2],c7[1],c8[1],c9[1],c10[1],
               c7[0],c8[0],c9[0],c10[0],c3[2],c3[3],c3[4],p[15]};

    return new real[][] {s0,s1,s2,s3};
  }

  triple[][] splitpatch4(triple[] p) {
    // Find new control points.
    triple[] c0=split(p[0],p[1],p[2],p[3]);
    triple[] c1=split(p[4],p[5],p[6],p[7]);
    triple[] c2=split(p[8],p[9],p[10],p[11]);
    triple[] c3=split(p[12],p[13],p[14],p[15]);

    triple[] c4=split(p[12],p[8],p[4],p[0]);
    triple[] c5=split(c3[0],c2[0],c1[0],c0[0]);
    triple[] c6=split(c3[1],c2[1],c1[1],c0[1]);
    triple[] c7=split(c3[2],c2[2],c1[2],c0[2]);

    triple[] c8=split(c3[3],c2[3],c1[3],c0[3]);
    triple[] c9=split(c3[4],c2[4],c1[4],c0[4]);
    triple[] c10=split(p[15],p[11],p[7],p[3]);

    // Set up 4 Bezier subpatches.
    triple[] s0={c4[2],c5[2],c6[2],c7[2],c4[1],c5[1],c6[1],c7[1],
                 c4[0],c5[0],c6[0],c7[0],p[12],c3[0],c3[1],c3[2]};
    triple[] s1={p[0],c0[0],c0[1],c0[2],c4[4],c5[4],c6[4],c7[4],
                 c4[3],c5[3],c6[3],c7[3],c4[2],c5[2],c6[2],c7[2]};
    triple[] s2={c0[2],c0[3],c0[4],p[3],c7[4],c8[4],c9[4],c10[4],
                 c7[3],c8[3],c9[3],c10[3],c7[2],c8[2],c9[2],c10[2]};
    triple[] s3={c7[2],c8[2],c9[2],c10[2],c7[1],c8[1],c9[1],c10[1],
                 c7[0],c8[0],c9[0],c10[0],c3[2],c3[3],c3[4],p[15]};

    return new triple[][] {s0,s1,s2,s3};
  }

  real bound(real[] p, real m(...real[]), real bound=p[0], int depth=maxdepth) {
    bound=m(bound,p[0],p[3],p[12],p[15]);
    if(m(-1,1)*(bound-m(p[1],p[2],p[4],p[5],p[6],p[7],p[8],
                        p[9],p[10],p[11],p[13],p[14])) >= 0)
      return bound;

    if(depth == 0) return p[0];
    --depth;
    real[][] s=splitpatch4(p);

    return m(bound(s[0],m,bound,depth),bound(s[1],m,bound,depth),
             bound(s[2],m,bound,depth),bound(s[3],m,bound,depth));
  }

  real bound(triple[] p, real m(...real[]), real f(triple), real bound=f(p[0]),
             int depth=maxdepth) {
    bound=m(bound,f(p[0]),f(p[3]),f(p[12]),f(p[15]));
    if(m(-1,1)*(bound-m(f(p[1]),f(p[2]),f(p[4]),f(p[5]),f(p[6]),f(p[7]),f(p[8]),
                        f(p[9]),f(p[10]),f(p[11]),f(p[13]),f(p[14]))) >= 0)
      return bound;

    if(depth == 0) return f(p[0]);
    --depth;
    triple[][] s=splitpatch4(p);

    return m(bound(s[0],m,f,bound,depth),bound(s[1],m,f,bound,depth),
             bound(s[2],m,f,bound,depth),bound(s[3],m,f,bound,depth));
  }

  triple bound(real m(...real[]), triple bound) {
    real x=bound(new real[] {P[0][0].x,P[0][1].x,P[0][2].x,P[0][3].x,
                             P[1][0].x,P[1][1].x,P[1][2].x,P[1][3].x,
                             P[2][0].x,P[2][1].x,P[2][2].x,P[2][3].x,
                             P[3][0].x,P[3][1].x,P[3][2].x,P[3][3].x},
      m,bound.x);
    real y=bound(new real[] {P[0][0].y,P[0][1].y,P[0][2].y,P[0][3].y,
                             P[1][0].y,P[1][1].y,P[1][2].y,P[1][3].y,
                             P[2][0].y,P[2][1].y,P[2][2].y,P[2][3].y,
                             P[3][0].y,P[3][1].y,P[3][2].y,P[3][3].y},
      m,bound.y);
    real z=bound(new real[] {P[0][0].z,P[0][1].z,P[0][2].z,P[0][3].z,
                             P[1][0].z,P[1][1].z,P[1][2].z,P[1][3].z,
                             P[2][0].z,P[2][1].z,P[2][2].z,P[2][3].z,
                             P[3][0].z,P[3][1].z,P[3][2].z,P[3][3].z},
      m,bound.z);
    return (x,y,z);
  }

  pair bound(real m(...real[]), projection Q, pair bound=project(P[0][0],Q)) {
    real x=bound(new triple[] {P[0][0],P[0][1],P[0][2],P[0][3],
			       P[1][0],P[1][1],P[1][2],P[1][3],
			       P[2][0],P[2][1],P[2][2],P[2][3],
			       P[3][0],P[3][1],P[3][2],P[3][3]},
      m,new real(triple v) {return project(v,Q).x;});
    real y=bound(new triple[] {P[0][0],P[0][1],P[0][2],P[0][3],
			       P[1][0],P[1][1],P[1][2],P[1][3],
			       P[2][0],P[2][1],P[2][2],P[2][3],
			       P[3][0],P[3][1],P[3][2],P[3][3]},
      m,new real(triple v) {return project(v,Q).y;});
    return (x,y);
  }

  triple min3,max3;
  bool havemin3,havemax3;

  pair min2,max2;
  bool havemin2,havemax2;

  void init() {
    havemin3=false;
    havemax3=false;
    havemin2=false;
    havemax2=false;
  }

  triple min(triple bound=P[0][0]) {
    if(havemin3) return min3;
    havemin3=true;
    return min3=bound(min,bound);
  }

  triple max(triple bound=P[0][0]) {
    if(havemax3) return max3;
    havemax3=true;
    return max3=bound(max,bound);
  }

  pair min(projection P, pair bound=project(this.P[0][0],P)) {
    if(havemin2) return min2;
    havemin2=true;
    return min2=bound(min,P,bound);
  }

  pair max(projection P, pair bound=project(this.P[0][0],P)) {
    if(havemax2) return max2;
    havemax2=true;
    return max2=bound(max,P,bound);
  }

  void operator init(triple[][] P) {
    init();
    this.P=copy(P);
  }

  void operator init(triple[] P) {
    init();
    this.P=new triple[][] {{P[0],P[1],P[2],P[3]},
                           {P[4],P[5],P[6],P[7]},
                           {P[8],P[9],P[10],P[11]},
                           {P[12],P[13],P[14],P[15]}};
  }

  void operator init(path3 external, triple[] internal=new triple[]) {
    if(!cyclic(external) || length(external) != 4)
      abortcyclic();
    P=new triple[4][4];
    init();
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

  void operator init(explicit guide3 external, triple[] internal=new triple[]) {
    operator init((path3) external,internal);
  }
}

struct surface {
  patch[] s;
  
  void operator init(patch s) {
    this.s=new patch[] {s};
  }

  void operator init(triple[][] P) {
    s=new patch[] {patch(P)};
  }

  void operator init(triple[][][] P) {
    s=sequence(new patch(int i) {return patch(P[i]);},s.length);
  }

  void operator init(path3 external, triple[] internal=new triple[]) {
    s=new patch[] {patch(external,internal)};
  }

  // A constructor for a (possibly) nonconvex cyclic path of length 4 that
  // returns an array of one or two surfaces in a given plane.
  void operator init (explicit path g, triple plane(pair)=XYplane) {
    if(!cyclic(g) || length(g) != 4)
      abortcyclic();
    for(int i=0; i < 4; ++i) {
      pair z=point(g,i);
      int w=windingnumber(subpath(g,i+1,i+3)--cycle,z);
      if(w != 0 && w != undefined) {
        pair w=point(g,i+2);
        real[][] T=intersections(z--w,g);
        path c,d;
        if(T.length > 2) {
          real t=T[1][1];
          real s=t-i;
          if(s < -1) s += 4;
          else if(s > 3) s -= 4;
          path close(path p, pair m) {
            return length(p) == 3 ? p--cycle : p--0.5*(m+point(g,t))--cycle;
          }
          if(s < 1) {
            c=close(subpath(g,i+s,i+2),w);
            d=close(subpath(g,i-2,i+s),w);
          } else {
            c=close(subpath(g,i+s,i+4),z);
            d=close(subpath(g,i,i+s),z);
          }
        } else {
          pair m=0.5*(z+w);
          c=subpath(g,i-2,i)--m--cycle;
          d=subpath(g,i,i+2)--m--cycle;
        }
        s=new patch[] {patch(path3(c,plane)),patch(path3(d,plane))};
        return;
      }
    }
    s=new patch[] {patch(path3(g,plane))};
  }

  void operator init (explicit guide g) {
    operator init((path) g);
  }

  void operator init(explicit path[] g, triple plane(pair)=XYplane) {
    for(int i=0; i < g.length; ++i)
      s.append(surface(g[i],plane).s);
  }
}

patch operator * (transform3 t, patch s)
{ 
  patch S;
  S.P=new triple[4][4];
  for(int i=0; i < s.P.length; ++i) { 
    triple[] si=s.P[i];
    triple[] Si=S.P[i];
    for(int j=0; j < si.length; ++j) { 
      Si[j]=t*si[j]; 
    } 
  }
  return S;
}
 
surface operator * (transform3 t, surface s)
{ 
  surface S;
  S.s=new patch[s.s.length];
  for(int i=0; i < s.s.length; ++i)
    S.s[i]=t*s.s[i];
  return S;
}

patch operator cast(triple[][] P)
{
  return patch(P);
}

path3[] bbox3(patch s)
{
  return box(s.min(),s.max());
}

private string nullsurface="null surface";

triple min(surface s)
{
  if(s.s.length == 0)
    abort(nullsurface);
  triple bound=s.s[0].min();
  for(int i=1; i < s.s.length; ++i)
    bound=s.s[i].min(bound);
  return bound;
}
  
triple max(surface s)
{
  if(s.s.length == 0)
    abort(nullsurface);
  triple bound=s.s[0].max();
  for(int i=1; i < s.s.length; ++i)
    bound=s.s[i].max(bound);
  return bound;
}

pair min(surface s, projection P)
{
  if(s.s.length == 0)
    abort(nullsurface);
  pair bound=s.s[0].min(P);
  for(int i=1; i < s.s.length; ++i)
    bound=s.s[i].min(P,bound);
  return bound;
}
  
pair max(surface s, projection P)
{
  if(s.s.length == 0)
    abort(nullsurface);
  pair bound=s.s[0].max(P);
  for(int i=1; i < s.s.length; ++i)
    bound=s.s[i].max(P,bound);
  return bound;
}

patch subpatchu(patch s, real ua, real ub)
{
  path3 G=s.uequals(ua)&subpath(s.vequals(1),ua,ub)&reverse(s.uequals(ub));
  path3 w=subpath(s.vequals(0),ub,ua);
  path3 i1=s.P[0][1]..controls s.P[1][1] and s.P[2][1]..s.P[3][1];
  path3 i2=s.P[0][2]..controls s.P[1][2] and s.P[2][2]..s.P[3][2];
  path3 s1=subpath(i1,ua,ub);
  path3 s2=subpath(i2,ua,ub);
  return patch(G..controls postcontrol(w,0) and precontrol(w,1)..cycle,
	       new triple[] {postcontrol(s1,0),postcontrol(s2,0),
		   precontrol(s2,1),precontrol(s1,1)});
}

patch subpatchv(patch s, real va, real vb)
{
  path3 G=subpath(s.uequals(0),va,vb)&s.vequals(vb)&subpath(s.uequals(1),vb,va);
  path3 w=s.vequals(va);
  path3 j1=s.P[1][0]..controls s.P[1][1] and s.P[1][2]..s.P[1][3];
  path3 j2=s.P[2][0]..controls s.P[2][1] and s.P[2][2]..s.P[2][3];
  path3 t1=subpath(j1,va,vb);
  path3 t2=subpath(j2,va,vb);

  return patch(G..controls precontrol(w,1) and postcontrol(w,0)..cycle,
	       new triple[] {postcontrol(t1,0),precontrol(t1,1),
		   precontrol(t2,1),postcontrol(t2,0)});
}

patch subpatch(patch s, real ua, real ub, real va, real vb)
{
  return subpatchu(subpatchv(s,va,vb),ua,ub);
}

triple point(patch s, real u, real v)
{
  return point(s.uequals(u),v);
}

void draw(frame f, patch s, pen p=currentpen)
{
  draw(f,s.P,p,s.min(),s.max());
}

void draw(frame f, surface s, pen p=currentpen)
{
  for(int i=0; i < s.s.length; ++i)
    draw(f,s.s[i],p);
}

void tensorshade(picture pic=currentpicture, patch s,
                 pen surfacepen=lightgray, light light=currentlight,
                 projection P=currentprojection, int ninterpolate=1)
{
  path[] b=box(s.min(P),s.max(P));
  tensorshade(pic,box(min(b),max(b)),surfacepen,s.colors(surfacepen,light),
              project(s.external(),P,1),project(s.internal(),P));
}

void draw(picture pic=currentpicture, surface s, int nu=nmesh, int nv=nu,
          pen surfacepen=lightgray, pen meshpen=nullpen,
          light light=currentlight, projection P=currentprojection)
{
  // Draw a mesh in the absence of lighting (override with meshpen=invisible).
  if(light.source == O && meshpen == nullpen) meshpen=currentpen;

  if(surfacepen != nullpen) {
    triple camera=P.camera;
    triple m=min(s);
    triple M=max(s);
    if(P.infinity)
      camera *= max(abs(m),abs(M));

    if(prc()) {
      pic.add(new void(frame f, transform3 t) {
          for(int i=0; i < s.s.length; ++i)
            draw(f,t*s.s[i],surfacepen);
        },true);
      if(s.s.length > 0) {
        pic.addPoint(m);
        pic.addPoint(M);
      }
      pic.is3D=true;
    } else {
      // Sort patches by mean distance from camera
      triple camera=P.camera;
      if(P.infinity)
        camera *= max(abs(min(s)),abs(max(s)));

      real[][] depth;
    
      for(int i=0; i < s.s.length; ++i) {
        triple[][] P=s.s[i].P;
        for(int j=0; j < nv; ++j) {
          real d=abs(camera-0.25*(P[0][0]+P[0][3]+P[3][3]+P[3][0]));
          depth.push(new real[] {d,i,j});
        }
      }

      depth=sort(depth);

      // Draw from farthest to nearest
      while(depth.length > 0) {
        real[] a=depth.pop();
        int i=round(a[1]);
        int j=round(a[2]);
        tensorshade(pic,s.s[i],surfacepen,light,P);
      }
    }
  }
    
  if(meshpen != nullpen) {
    for(int k=0; k < s.s.length; ++k) {
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
        draw(pic,s.s[k].uequals(i*step),meshpen);
    
      real step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
        draw(pic,s.s[k].vequals(j*step),meshpen);
    }
  }
}

void draw(picture pic=currentpicture, triple[][][] P, pen p=currentpen)
{
  for(int i=0; i < P.length; ++i)
    draw(pic,surface(P[i]),p);
}

surface extrude(path g, triple elongation=Z)
{
  patch[] allocate;
  surface S;
  path3 G=path3(g);
  path3 G2=shift(elongation)*G;
  S.s=sequence(new patch(int i) {
      return patch(subpath(G,i,i+1)--subpath(G2,i+1,i)--cycle);
    },length(G));
  return S;
}

private transform3 shift(transform3 T, triple v)
{
  transform3 t=copy(T);
  t[0][3] += v.x;
  t[1][3] += v.y;
  t[2][3] += v.z;
  return t;
}

private transform3 identity4=identity(4);

void label(frame f, string s, transform t=identity(), transform3 T=identity4,
           triple position, pair align=0, pen p=currentpen)
{
  draw(f,shift(T,position)*surface(bezulate(texpath(s,t,0,align,p))),p);
}

void label(picture pic=currentpicture, string s, transform t=identity(),
           transform3 T=identity4, triple position, pair align=0,
           pen p=currentpen)
{
  draw(pic,shift(T,position)*surface(bezulate(texpath(s,t,0,align,p))),p);
}

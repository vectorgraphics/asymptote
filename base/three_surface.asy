import graph_settings;
import bezulate;

private real Fuzz=10.0*realEpsilon;

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

    struct dir {
      triple post;
      triple pre;
      void operator init(triple z0, triple c0, triple c1, triple z1) {
	real epsilon=Fuzz*abs(z0-z1);

	post=c0-z0;
	if(abs(post) > epsilon) post=unit(post);
	else {
	  post=z0-2*c0+c1;
	  if(abs(post) > epsilon) post=unit(post);
	  else post=unit(z1-z0+3*(c0-c1));
	}
	
	pre=z1-c1;
	if(abs(pre) > epsilon) pre=unit(pre);
	else {
	  pre=2*c1-c0-z1;
	  if(abs(pre) > epsilon) pre=unit(pre);
	  else pre=unit(z1-z0+3*(c0-c1));
	}
      }
    }

    dir dir0=dir(P[0][0],P[0][1],P[0][2],P[0][3]);
    dir dir1=dir(P[0][3],P[1][3],P[2][3],P[3][3]);
    dir dir2=dir(P[3][3],P[3][2],P[3][1],P[3][0]);
    dir dir3=dir(P[3][0],P[2][0],P[1][0],P[0][0]);

    return new pen[] {color(dir3.pre,-dir0.post),
	color(-dir1.post,-dir0.pre),
	color(-dir1.pre,dir2.post),
	color(dir3.post,dir2.pre)};
  }

  triple bound(real m(real[], real), triple b) {
    real x=m(new real[] {P[0][0].x,P[0][1].x,P[0][2].x,P[0][3].x,
			 P[1][0].x,P[1][1].x,P[1][2].x,P[1][3].x,
			 P[2][0].x,P[2][1].x,P[2][2].x,P[2][3].x,
			 P[3][0].x,P[3][1].x,P[3][2].x,P[3][3].x},b.x);
    real y=m(new real[] {P[0][0].y,P[0][1].y,P[0][2].y,P[0][3].y,
			 P[1][0].y,P[1][1].y,P[1][2].y,P[1][3].y,
			 P[2][0].y,P[2][1].y,P[2][2].y,P[2][3].y,
			 P[3][0].y,P[3][1].y,P[3][2].y,P[3][3].y},b.y);
    real z=m(new real[] {P[0][0].z,P[0][1].z,P[0][2].z,P[0][3].z,
			 P[1][0].z,P[1][1].z,P[1][2].z,P[1][3].z,
			 P[2][0].z,P[2][1].z,P[2][2].z,P[2][3].z,
			 P[3][0].z,P[3][1].z,P[3][2].z,P[3][3].z},b.z);
    return (x,y,z);
  }

  pair bound(real m(triple[], real f(triple), real),
	     projection P, pair b=project(this.P[0][0],P)) {
    triple[] Q=controlpoints();
    project P=P.transform3();
    return (m(Q,new real(triple v) {return P(v).x;},b.x),     
	    m(Q,new real(triple v) {return P(v).y;},b.y));
  }

  pair min2,max2;
  bool havemin2,havemax2;

  triple min3,max3;
  bool havemin3,havemax3;

  void init() {
    havemin2=false;
    havemax2=false;
    havemin3=false;
    havemax3=false;
  }

  triple min(triple bound=P[0][0]) {
    if(havemin3) return minbound(min3,bound);
    havemin3=true;
    return min3=bound(minbound,bound);
  }

  triple max(triple bound=P[0][0]) {
    if(havemax3) return maxbound(max3,bound);
    havemax3=true;
    return max3=bound(maxbound,bound);
  }

  pair min(projection P, pair bound=project(this.P[0][0],P)) {
    if(havemin2) return minbound(min2,bound);
    havemin2=true;
    return min2=bound(minbound,P,bound);
  }

  pair max(projection P, pair bound=project(this.P[0][0],P)) {
    if(havemax2) return maxbound(max2,bound);
    havemax2=true;
    return max2=bound(maxbound,P,bound);
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
  
  void operator init(...patch[] s) {
    this.s=s;
  }

  void operator init(triple[][] P) {
    s=new patch[] {patch(P)};
  }

  void operator init(triple[][][] P) {
    s=sequence(new patch(int i) {return patch(P[i]);},P.length);
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

void drawprc(frame f, patch s, pen p=currentpen)
{
  draw(f,s.P,p,s.min(),s.max());
}

void tensorshade(frame f, patch s, pen surfacepen=lightgray,
		 light light=currentlight,
		 projection P=currentprojection, int ninterpolate=1)
{
  tensorshade(f,box(s.min(P),s.max(P)),surfacepen,s.colors(surfacepen,light),
	      project(s.external(),P,1),project(s.internal(),P));
}

void draw(frame f, surface s, int nu=nmesh, int nv=nu,
	  pen surfacepen=lightgray, pen meshpen=nullpen,
	  light light=currentlight,
	  transform3 t=identity4, projection P=currentprojection)
{
  // Draw a mesh in the absence of lighting (override with meshpen=invisible).
  if(light.source == O && meshpen == nullpen) meshpen=currentpen;

  if(prc()) {
    for(int i=0; i < s.s.length; ++i)
      drawprc(f,t*s.s[i],surfacepen);
  } else {
    if(surfacepen != nullpen) {
      projection P=t*P;
      triple m=min(s);
      triple M=max(s);

      // Sort patches by mean distance from camera
      triple camera=P.camera;
      if(P.infinity)
	camera *= max(abs(m),abs(M));

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
	tensorshade(f,s.s[i],surfacepen,light,P);
      }
    }
  }
    
  if(meshpen != nullpen) {
    for(int k=0; k < s.s.length; ++k) {
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
	draw(f,s.s[k].uequals(i*step),meshpen,t,P);
    
      real step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
	draw(f,s.s[k].vequals(j*step),meshpen,t,P);
    }
  }
}

void draw(picture pic=currentpicture, surface s, int nu=nmesh, int nv=nu,
	  pen surfacepen=lightgray, pen meshpen=nullpen,
	  light light=currentlight, projection P=currentprojection)
{
  if(s.s.length == 0) return;
  pic.is3D=true;
  pic.add(new void(frame f, transform3 t) {
      draw(f,s,nu,nv,surfacepen,meshpen,light,t,P);
    },true);
  pic.addPoint(min(s));
  pic.addPoint(max(s));
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

void label(frame f, Label L, triple position, align align=NoAlign,
	   pen p=currentpen, light light=nolight,
	   projection P=currentprojection)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  path[] g=texpath(L.s,0,L.align.dir,L.p);
  if(prc())
    draw(f,shift(position)*L.T3*surface(bezulate(g)),L.p);
  else
    fill(f,project(L.T3*path3(g),P),light.intensity(L.T3*Z)*L.p);
}

void label(picture pic=currentpicture, Label L, triple position,
	   align align=NoAlign, pen p=currentpen, light light=nolight,
	   projection P=currentprojection)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  path[] g=texpath(L.s,0,L.align.dir,L.p);
  if(g.length == 0) return;
  pic.is3D=true;
  if(prc()) {
    surface s=L.T3*surface(bezulate(g));
    pic.add(new void(frame f, transform3 t) {
	draw(f,shift(t*position)*s,L.p);
      },true);
    pic.addPoint(position,min(s));
    pic.addPoint(position,max(s));
  } else {
    pic.add(new void(frame f, transform3 t) {
	fill(f,project(shift(position)*inverse(t)*L.T3*path3(g),t*P),
	     light.intensity(L.T3*Z)*L.p);
      },true);
    pic.addPoint(position,L.T3*XYplane(min(g)));
    pic.addPoint(position,L.T3*XYplane(max(g)));
  }
}

void label(picture pic=currentpicture, Label L, path3 g,
           align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  Label L=Label(L,align,p,filltype);
  bool relative=L.position.relative;
  real position=L.position.position.x;
  pair Align=L.align.dir;
  bool alignrelative=L.align.relative;
  if(L.defaultposition) {relative=true; position=0.5;}
  if(relative) position=reltime(g,position);
  if(L.align.default) {
    alignrelative=true;
    Align=position <= 0 ? S : position >= length(g) ? N : E;
  }
  label(pic,L,point(g,position),
	alignrelative ? -Align*xypart(inverse(L.T3)*dir(g,position))*I : Align);
}

private real a=4/3*(sqrt(2)-1);
private transform3 t=rotate(90,O,Z);
private transform3 i=zscale3(-1);

private patch octant1=patch(X{Z}..{-X}Z..Z{Y}..{-Z}Y{X}..{-Y}cycle,
			    new triple[] {(1,a,a),(a,a^2,1),(a^2,a,1),(a,1,a)});
private patch octant2=t*octant1;
private patch octant3=t*octant2;
private patch octant4=t*octant3;

restricted surface unitsphere=surface(octant1,octant2,octant3,octant4,
				      i*octant1,i*octant2,i*octant3,i*octant4);

void dot(frame f, triple v, pen p=currentpen, filltype filltype=Fill,
	 transform3 t=identity4, projection P=currentprojection)
{
  if(prc())
    draw(f,shift(t*v)*scale3(0.5*dotsize(p))*unitsphere,p);
  else dot(f,project(v,t*P),p,filltype);
};

void dot(picture pic=currentpicture, triple v, pen p=currentpen,
	 filltype filltype=Fill, projection P=currentprojection)
{
  pic.is3D=true;
  pic.add(new void(frame f, transform3 t) {
      dot(f,v,p,filltype,t,P);
    },true);
  triple R=0.5*dotsize(p)*(1,1,1);
  pic.addBox(v,v,-R,R);
}

void dot(picture pic=currentpicture, triple[] v, pen p=currentpen,
	 filltype filltype=Fill)
{
  for(int i=0; i < v.length; ++i) dot(pic,v[i],p,filltype);
}

void dot(picture pic=currentpicture, explicit path3 g, pen p=currentpen,
	 filltype filltype=Fill)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p,filltype);
}

void dot(picture pic=currentpicture, explicit guide3 g, pen p=currentpen,
	 filltype filltype=Fill)
{
  dot(pic,(path3) g,p,filltype);
}

void dot(picture pic=currentpicture, explicit path3[] g, pen p=currentpen,
	 filltype filltype=Fill)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p,filltype);
}

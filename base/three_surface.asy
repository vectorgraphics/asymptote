import bezulate;

private real Fuzz=10.0*realEpsilon;

private void abortcyclic() {abort("cyclic path of length 4 expected");}

struct patch {
  triple[][] P=new triple[4][4];
  pen[] colors; // Optionally associate specific colors to this patch.
  bool straight;

  path3 external() {
    return
      P[0][0]..controls P[0][1] and P[0][2]..
      P[0][3]..controls P[1][3] and P[2][3]..
      P[3][3]..controls P[3][2] and P[3][1]..
      P[3][0]..controls P[2][0] and P[1][0]..cycle;
  }

  triple[] internal() {
    return new triple[] {P[1][1],P[1][2],P[2][2],P[2][1]};
  }

  triple[] controlpoints() {
    return new triple[] {
      P[0][0],P[0][1],P[0][2],P[0][3],
	P[1][0],P[1][1],P[1][2],P[1][3],
	P[2][0],P[2][1],P[2][2],P[2][3],
	P[3][0],P[3][1],P[3][2],P[3][3]};
  }

  triple Bu(int j, real u) {return bezier(P[0][j],P[1][j],P[2][j],P[3][j],u);}
  triple BuP(int j, real u) {return bezierP(P[0][j],P[1][j],P[2][j],P[3][j],u);}

  path3 uequals(real u) {
    return straight ? Bu(0,u)--Bu(3,u) :
      Bu(0,u)..controls Bu(1,u) and Bu(2,u)..Bu(3,u);
  }

  triple Bv(int i, real v) {return bezier(P[i][0],P[i][1],P[i][2],P[i][3],v);}
  triple BvP(int i, real v) {return bezierP(P[i][0],P[i][1],P[i][2],P[i][3],v);}

  path3 vequals(real v) {
    return straight ? Bv(0,v)--Bv(3,v) :
      Bv(0,v)..controls Bv(1,v) and Bv(2,v)..Bv(3,v);
  }

  triple point(real u, real v) {	
    return bezier(Bu(0,u),Bu(1,u),Bu(2,u),Bu(3,u),v);
  }

  triple normal(real u, real v) {
    return cross(bezier(BuP(0,u),BuP(1,u),BuP(2,u),BuP(3,u),v),   
		 bezier(BvP(0,v),BvP(1,v),BvP(2,v),BvP(3,v),u));
  }

  pen[] colors(pen surfacepen=lightgray, light light=currentlight,
	       bool outward=false, projection Q=null) {
    if(colors.length != 0)
      return colors;
    pen color(real u, real v) {
      triple n=normal(u,v);
      if(!outward)
	n *= sgn(dot(n,Q.vector()));
      return light.intensity(n)*surfacepen;
    }

    return new pen[] {color(0,0),color(0,1),color(1,1),color(1,0)};
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

  pair bound(real m(triple[], real f(triple), real), projection P,
	     pair b=project(this.P[0][0],P)) {
    triple[] Q=controlpoints();
    transform3 t=P.t;
    return (m(Q,new real(triple v) {return project(v,t).x;},b.x),
	    m(Q,new real(triple v) {return project(v,t).y;},b.y));
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
    straight=false;
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

  pair min(projection P, pair bound=project(this.P[0][0],P.t)) {
    if(havemin2) return minbound(min2,bound);
    havemin2=true;
    return min2=bound(minbound,P,bound);
  }

  pair max(projection P, pair bound=project(this.P[0][0],P.t)) {
    if(havemax2) return maxbound(max2,bound);
    havemax2=true;
    return max2=bound(maxbound,P,bound);
  }

  void operator init(triple[][] P, pen[] colors=new pen[], bool straight=false)
  {
    init();
    this.P=copy(P);
    if(colors.length != 0)
      this.colors=copy(colors);
    this.straight=straight;
  }

  void operator init(patch s) {
    operator init(s.P,s.colors,s.straight);
  }
  
  void operator init(path3 external, triple[] internal=new triple[],
		     pen[] colors=new pen[]) {
    if(colors.length != 0)
      this.colors=copy(colors);

    if(!cyclic(external) || length(external) != 4)
      abortcyclic();

    init();

    if(internal.length == 0) {
      if(piecewisestraight(external)) straight=true;

      internal=new triple[4];
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
}

  struct surface {
    patch[] s;
  
    bool empty() {
      return s.length == 0;
    }

    void operator init(int n) {
      s=new patch[n];
    }

    void operator init(... patch[] s) {
      this.s=s;
    }

    void operator init(surface s) {
      this.s=new patch[s.s.length];
      for(int i=0; i < s.s.length; ++i)
	this.s[i]=patch(s.s[i]);
    }

    void operator init(triple[][] P, pen[] colors=new pen[]) {
      s=new patch[] {patch(P,colors)};
    }

    void operator init(triple[][][] P, pen[][] colors=new pen[][]) {
      s=sequence(new patch(int i) {
	  return patch(P[i],colors.length == 0 ? new pen[] : colors[i]);
	},P.length);
    }

    void operator init(path3 external, triple[] internal=new triple[],
		       pen[] colors=new pen[]) {
      s=new patch[] {patch(external,internal,colors)};
    }

    void push(path3 external, triple[] internal=new triple[],
	      pen[] colors=new pen[]) {
      s.push(patch(external,internal,colors));
    }

    // A constructor for a (possibly) nonconvex cyclic path of length 4 that
    // returns an array of one or two surfaces in a given plane.
    void operator init (path g, triple plane(pair)=XYplane) {
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

    void operator init(explicit path[] g, triple plane(pair)=XYplane) {
      for(int i=0; i < g.length; ++i)
	s.append(surface(g[i],plane).s);
    }

    void push(patch s) {
      this.s.push(s);
    }

    void append(surface s) {
      this.s.append(s.s);
    }
  }

    patch operator * (transform3 t, patch s)
{ 
  patch S;
  for(int i=0; i < 4; ++i) { 
    triple[] si=s.P[i];
    triple[] Si=S.P[i];
    for(int j=0; j < 4; ++j) { 
      Si[j]=t*si[j]; 
    }
  }
  S.colors=copy(s.colors);
  S.straight=s.straight;
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
  return s.point(u,v);
}

void draw3D(frame f, patch s, material m=lightgray, light light=currentlight,
	    bool localsub=false)
{
  if(!light.on) m=emissive(m.p[0],m.granularity);
  real granularity=m.granularity >= 0 ? m.granularity : defaultgranularity;
  draw(f,s.P,m.p,m.opacity,m.shininess,granularity,localsub,s.min(),s.max());
}

void tensorshade(transform t=identity(), frame f, patch s, bool outward=false,
		 pen surfacepen=lightgray, light light=currentlight,
		 projection P)
{
  tensorshade(f,box(t*s.min(P),t*s.max(P)),surfacepen,
	      s.colors(surfacepen,light,outward,P),t*project(s.external(),P,1),
	      t*project(s.internal(),P));
}

void draw(transform t=identity(), frame f, surface s, int nu=1, int nv=1,
	  bool outward=false, material surfacepen=lightgray,
	  pen meshpen=nullpen, light light=currentlight,
	  projection P=currentprojection)
{
  bool mesh=meshpen != nullpen;

  if(is3D()) {
    for(int i=0; i < s.s.length; ++i)
      draw3D(f,s.s[i],surfacepen,light);
    if(mesh) {
      for(int k=0; k < s.s.length; ++k) {
	real step=nu == 0 ? 0 : 1/nu;
	for(int i=0; i <= nu; ++i)
	  draw(f,s.s[k].uequals(i*step),thin+meshpen);
	step=nv == 0 ? 0 : 1/nv;
	for(int j=0; j <= nv; ++j)
	  draw(f,s.s[k].vequals(j*step),thin+meshpen);
      }
    }
  } else {
    bool surface=surfacepen != nullpen;
    begingroup(f);
    // Sort patches by mean distance from camera
    triple camera=P.camera;
    if(P.infinity)
      camera=P.target+camerafactor*max(abs(min(s)),abs(max(s)))*
	unit(P.vector());

    real[][] depth;
    
    for(int i=0; i < s.s.length; ++i) {
      triple[][] P=s.s[i].P;
      real d=abs(camera-0.25*(P[0][0]+P[0][3]+P[3][3]+P[3][0]));
      depth.push(new real[] {d,i});
    }

    depth=sort(depth);

    // Draw from farthest to nearest
    while(depth.length > 0) {
      real[] a=depth.pop();
      int i=round(a[1]);
      if(surface)
	tensorshade(t,f,s.s[i],outward,surfacepen.p[0],light,P);
      if(mesh)
	draw(f,project(s.s[i].external(),P),meshpen);
    }
    endgroup(f);
  }
}

void draw(picture pic=currentpicture, surface s, int nu=1, int nv=1,
	  bool outward=false, material surfacepen=lightgray,
	  pen meshpen=nullpen, light light=currentlight)
{
  if(s.empty()) return;

  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      surface S=t*s;
      if(is3D()) {
	draw(f,S,nu,nv,surfacepen,meshpen,light);
      } else if(pic != null)
	pic.add(new void(frame f, transform T) {
	    draw(T,f,S,nu,nv,outward,surfacepen,meshpen,light,P);
	  },true);
      if(pic != null) {
	pic.addPoint(min(S,P));
	pic.addPoint(max(S,P));
      }
    },true);
  pic.addPoint(min(s));
  pic.addPoint(max(s));

  if(meshpen != nullpen) {
    if(is3D()) meshpen=thin+meshpen;
    for(int k=0; k < s.s.length; ++k) {
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
	addPath(pic,s.s[k].uequals(i*step),meshpen);
      step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
	addPath(pic,s.s[k].vequals(j*step),meshpen);
    }
  }
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

triple rectify(triple dir) 
{
  real scale=max(abs(dir.x),abs(dir.y),abs(dir.z));
  if(scale != 0) dir *= 0.5/scale;
  dir += (0.5,0.5,0.5);
  return dir;
}

path[] align(path[] g, transform t=identity(), pair position,
	     pair align, pen p=currentpen)
{
  pair m=min(g);
  pair M=max(g);
  pair dir=rectify(inverse(t)*-align);
  if(basealign(p) == 1)
    dir -= (0,m.y/(M.y-m.y));
  pair a=m+realmult(dir,M-m);
  return shift(position+align*labelmargin(p))*t*shift(-a)*g;
}

path3[] align(path3[] g, transform3 t=identity4, triple position,
	      triple align, pen p=currentpen)
{
  if(determinant(t) == 0) return g;
  triple m=min(g);
  triple M=max(g);
  triple dir=rectify(inverse(t)*-align);
  triple a=m+realmult(dir,M-m);
  return shift(position+align*labelmargin(p))*t*shift(-a)*g;
}

surface align(surface s, transform3 t=identity4, triple position,
	      triple align, pen p=currentpen)
{
  if(determinant(t) == 0) return s;
  triple m=min(s);
  triple M=max(s);
  triple dir=rectify(inverse(t)*-align);
  triple a=m+realmult(dir,M-m);
  return shift(position+align*labelmargin(p))*t*shift(-a)*s;
}

surface surface(Label L, triple position=O)
{
  path[] g=texpath(L);
  surface s=surface(bezulate(g));
  return L.align.is3D ? align(s,L.T3,position,L.align.dir3,L.p) :
    shift(position)*L.T3*s;
}

path[] path(Label L, pair z=0, projection P)
{
  path[] g=texpath(L);
  if(L.defaulttransform) {
    return L.align.is3D ? align(g,z,project(L.align.dir3,P)-project(O,P),L.p) :
      shift(z)*g;
  } else {
    path3[] G=path3(g);
    return L.align.is3D ? shift(z)*project(align(G,L.T3,O,L.align.dir3,L.p),P) :
      shift(z)*project(L.T3*G,P);
  }
}

void label(frame f, Label L, triple position, align align=NoAlign,
	   pen p=currentpen, light light=nolight,
	   projection P=currentprojection)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  if(L.defaulttransform)
    L.T3=transform3(P);
  if(is3D()) {
    for(patch S : surface(L,position).s)
      draw3D(f,S,L.p,light);
  } else fill(f,path(L,project(position,P.t),P),
	      light.intensity(L.T3*Z)*L.p);
}

void label(picture pic=currentpicture, Label L, triple position,
	   align align=NoAlign, pen p=currentpen, light light=nolight)
{
  Label L=L.copy();
  L.align(align);
  L.p(p);
  L.position(0);
  path[] g=texpath(L);
  if(g.length == 0) return;
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      triple v=t*position;
      if(L.defaulttransform)
	L.T3=transform3(P);
      if(is3D()) {
	for(patch S : surface(L,v).s)
	  draw3D(f,S,L.p,light);

      }
      if(pic != null)
	fill(project(v,P.t),pic,path(L,P),light.intensity(L.T3*Z)*L.p);
    },!L.defaulttransform);

  if(L.defaulttransform)
    L.T3=transform3(currentprojection);
  path3[] G=path3(g);
  G=L.align.is3D ? align(G,L.T3,O,L.align.dir3,L.p) : L.T3*G;
  pic.addBox(position,position,min(G),max(G));
}

void label(picture pic=currentpicture, Label L, path3 g, align align=NoAlign,
	   pen p=currentpen)
{
  Label L=Label(L,align,p);
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
	alignrelative ?
	-Align*project(dir(g,position),currentprojection.t)*I : L.align);
}

restricted surface nullsurface;

private real a=4/3*(sqrt(2)-1);
private transform3 t=rotate(90,O,Z);
private transform3 t2=t*t;
private transform3 t3=t2*t;
private transform3 i=xscale3(-1)*zscale3(-1);

private patch octant1=patch(X{Z}..{-X}Z..Z{Y}..{-Z}Y{X}..{-Y}cycle,
			    new triple[] {(1,a,a),(a,a^2,1),(a^2,a,1),(a,1,a)});

restricted surface unitsphere=surface(octant1,t*octant1,t2*octant1,t3*octant1,
				      i*octant1,i*t*octant1,i*t2*octant1,
				      i*t3*octant1);

private patch unitcone1=patch(X--Z--Z--Y{X}..{-Y}cycle,
			      new triple[] {(2/3,2/3*a,1/3),Z,Z,
					    (2/3*a,2/3,1/3)});

restricted surface unitcone=surface(unitcone1,t*unitcone1,t2*unitcone1,
				    t3*unitcone1);
restricted surface solidcone=surface(...unitcone.s);
solidcone.push(unitcircle3);

private patch unitcylinder1=patch(X--X+Z{Y}..{-X}Y+Z--Y{X}..{-Y}cycle);

restricted surface unitcylinder=surface(unitcylinder1,t*unitcylinder1,
					t2*unitcylinder1,t3*unitcylinder1);

private patch unitplane=patch(O--X--(X+Y)--Y--cycle);
restricted surface unitcube=surface(unitplane,
				    rotate(90,O,X)*unitplane,
				    rotate(-90,O,Y)*unitplane,
				    shift(Z)*unitplane,
				    rotate(90,X,X+Y)*unitplane,
				    rotate(-90,Y,X+Y)*unitplane);
restricted surface unitplane=surface(unitplane);
restricted surface unitdisk=surface(unitcircle3);

void dot(frame f, triple v, pen p=currentpen,
	 light light=nolight, projection P=currentprojection)
{
  if(is3D())
    for(patch s : unitsphere.s)
      draw3D(f,shift(v)*scale3(0.5*dotsize(p))*s,
	     material(p,granularity=dotgranularity),light);
  else dot(f,project(v,P.t),p);
}

void dot(frame f, path3 g, pen p=currentpen, projection P=currentprojection)
{
  for(int i=0; i <= length(g); ++i) dot(f,point(g,i),p,P);
}

void dot(frame f, path3[] g, pen p=currentpen, projection P=currentprojection)
{
  for(int i=0; i < g.length; ++i) dot(f,g[i],p,P);
}

void dot(picture pic=currentpicture, triple v, pen p=currentpen,
	 light light=nolight)
{
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      if(is3D())
	for(patch s : unitsphere.s)
	  draw3D(f,shift(t*v)*scale3(0.5*linewidth(dotsize(p)+p))*s,
		 material(p,granularity=dotgranularity),light);
      if(pic != null)
	dot(pic,project(t*v,P.t),p);
    },true);
  triple R=0.5*dotsize(p)*(1,1,1);
  pic.addBox(v,v,-R,R);
}

void dot(picture pic=currentpicture, triple[] v, pen p=currentpen)
{
  for(int i=0; i < v.length; ++i) dot(pic,v[i],p);
}

void dot(picture pic=currentpicture, path3 g, pen p=currentpen)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p);
}

void dot(picture pic=currentpicture, path3[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p);
}

void dot(picture pic=currentpicture, Label L, triple v, align align=NoAlign,
         string format=defaultformat, pen p=currentpen)
{
  Label L=L.copy();
  if(L.s == "") {
    if(format == "") format=defaultformat;
    L.s="("+format(format,v.x)+","+format(format,v.y)+","+
      format(format,v.z)+")";
  }
  L.align(align,E);
  L.p(p);
  dot(pic,v,p);
  label(pic,L,v);
}

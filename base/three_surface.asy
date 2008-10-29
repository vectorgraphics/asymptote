import bezulate;

int nslice=12;
real camerafactor=1.2;

private real Fuzz=10.0*realEpsilon;

struct patch {
  triple[][] P=new triple[4][4];
  triple[] normals; // Optionally specify 4 normal vectors at the corners.
  pen[] colors;     // Optionally specify 4 corner colors.
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
  triple BuPP(int j, real u) {
    return bezierPP(P[0][j],P[1][j],P[2][j],P[3][j],u);
  }
  triple BuPPP(int j) {return bezierPPP(P[0][j],P[1][j],P[2][j],P[3][j]);}

  path3 uequals(real u) {
    return straight ? Bu(0,u)--Bu(3,u) :
      Bu(0,u)..controls Bu(1,u) and Bu(2,u)..Bu(3,u);
  }

  triple Bv(int i, real v) {return bezier(P[i][0],P[i][1],P[i][2],P[i][3],v);}
  triple BvP(int i, real v) {return bezierP(P[i][0],P[i][1],P[i][2],P[i][3],v);}
  triple BvPP(int i, real v) {
    return bezierPP(P[i][0],P[i][1],P[i][2],P[i][3],v);
  }
  triple BvPPP(int i) {return bezierPPP(P[i][0],P[i][1],P[i][2],P[i][3]);}

  path3 vequals(real v) {
    return straight ? Bv(0,v)--Bv(3,v) :
      Bv(0,v)..controls Bv(1,v) and Bv(2,v)..Bv(3,v);
  }

  triple point(real u, real v) {        
    return bezier(Bu(0,u),Bu(1,u),Bu(2,u),Bu(3,u),v);
  }

// compute normal vectors for degenerate cases
  private triple normal0(real u, real v, real epsilon) {
    triple n=0.5*(cross(bezier(BuPP(0,u),BuPP(1,u),BuPP(2,u),BuPP(3,u),v),
			bezier(BvP(0,v),BvP(1,v),BvP(2,v),BvP(3,v),u))+
		  cross(bezier(BuP(0,u),BuP(1,u),BuP(2,u),BuP(3,u),v),   
			bezier(BvPP(0,v),BvPP(1,v),BvPP(2,v),BvPP(3,v),u)));
    return (abs(n) > epsilon) ? n :
      1/6*cross(bezier(BuPPP(0),BuPPP(1),BuPPP(2),BuPPP(3),v),
		bezier(BvP(0,v),BvP(1,v),BvP(2,v),BvP(3,v),u))+
      0.25*cross(bezier(BuPP(0,u),BuPP(1,u),BuPP(2,u),BuPP(3,u),v),   
                 bezier(BvPP(0,v),BvPP(1,v),BvPP(2,v),BvPP(3,v),u))+
      1/6*cross(bezier(BuP(0,u),BuP(1,u),BuP(2,u),BuP(3,u),v),   
                bezier(BvPPP(0),BvPPP(1),BvPPP(2),BvPPP(3),u))+
      1/12*(cross(bezier(BuPPP(0),BuPPP(1),BuPPP(2),BuPPP(3),v),
                  bezier(BvPP(0,v),BvPP(1,v),BvPP(2,v),BvPP(3,v),u))+
            cross(bezier(BuPP(0,u),BuPP(1,u),BuPP(2,u),BuPP(3,u),v),   
                  bezier(BvPPP(0),BvPPP(1),BvPPP(2),BvPPP(3),u)))+
      1/36*cross(bezier(BuPPP(0),BuPPP(1),BuPPP(2),BuPPP(3),v),   
                 bezier(BvPPP(0),BvPPP(1),BvPPP(2),BvPPP(3),u));
  }

  static real fuzz=1000*realEpsilon;

  triple normal(real u, real v) {
    triple n=cross(bezier(BuP(0,u),BuP(1,u),BuP(2,u),BuP(3,u),v),   
                   bezier(BvP(0,v),BvP(1,v),BvP(2,v),BvP(3,v),u));
    real epsilon=fuzz*change2(P);
    return (abs(n) > epsilon) ? n : normal0(u,v,epsilon);
  }
  
  triple normal00() {
    triple n=9*cross(P[1][0]-P[0][0],P[0][1]-P[0][0]);
    real epsilon=fuzz*change2(P);
    return abs(n) > epsilon ? n : normal0(0,0,epsilon);
  }

  triple normal01() {
    triple n=9*cross(P[1][3]-P[0][3],P[0][3]-P[0][2]);
    real epsilon=fuzz*change2(P);
    return abs(n) > epsilon ? n : normal0(0,1,epsilon);
  }

  triple normal11() {
    triple n=9*cross(P[3][3]-P[2][3],P[3][3]-P[3][2]);
    real epsilon=fuzz*change2(P);
    return abs(n) > epsilon ? n : normal0(1,1,epsilon);
  }

  triple normal10() {
    triple n=9*cross(P[3][0]-P[2][0],P[3][1]-P[3][0]);
    real epsilon=fuzz*change2(P);
    return abs(n) > epsilon ? n : normal0(1,0,epsilon);
  }

  pen[] colors(material m, light light=currentlight) {
    if(colors.length > 0) return colors;
    if(normals.length > 0)
      return new pen[] {light.color(normals[0],m),
	  light.color(normals[1],m),light.color(normals[2],m),
	  light.color(normals[3],m)};
    
    return new pen[] {light.color(normal00(),m),light.color(normal01(),m),
	light.color(normal11(),m),light.color(normal10(),m)};
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

  triple min3,max3;
  bool havemin3,havemax3;

  void init() {
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

  triple center() {
    return 0.5*(this.min()+this.max());
  }

  triple cornermean() {
    return 0.25*(P[0][0]+P[0][3]+P[3][3]+P[3][0]);
  }

  pair min(projection P, pair bound=project(this.P[0][0],P.t)) {
    return minbound(controlpoints(),P.t,bound);
  }

  pair max(projection P, pair bound=project(this.P[0][0],P.t)) {
    return maxbound(controlpoints(),P.t,bound);
  }

  void operator init(triple[][] P, triple[] normals=new triple[],
                     pen[] colors=new pen[], bool straight=false) {
    init();
    this.P=copy(P);
    if(normals.length != 0)
      this.normals=copy(normals);
    if(colors.length != 0)
      this.colors=copy(colors);
    this.straight=straight;
  }

  void operator init(patch s) {
    operator init(s.P,s.normals,s.colors,s.straight);
  }
  
  static real nineth=1/9;

  // A constructor for a convex cyclic path of length <= 4 with optional
  // arrays of 4 internal points, corner normals and pens.
  void operator init(path3 external, triple[] internal=new triple[],
                     triple[] normals=new triple[], pen[] colors=new pen[]) {
    init();
    int L=length(external);
    if(L > 4 || !cyclic(external))
      abort("cyclic path3 of length <= 4 expected");
    if(L == 1) {
      external=external--cycle--cycle--cycle;
      if(colors.length > 0) colors.append(array(3,colors[0]));
      if(normals.length > 0) normals.append(array(3,normals[0]));
    } else if(L == 2) {
      external=external--cycle--cycle;
      if(colors.length > 0) colors.append(array(2,colors[0]));
      if(normals.length > 0) normals.append(array(2,normals[0]));
    } else if(L == 3) {
      external=external--cycle;
      if(colors.length > 0) colors.push(colors[0]);
      if(normals.length > 0) normals.push(normals[0]);
    }
    if(normals.length != 0)
      this.normals=copy(normals);
    if(colors.length != 0)
      this.colors=copy(colors);

    if(internal.length == 0) {
      if(piecewisestraight(external)) straight=true;

      internal=new triple[4];
      for(int j=0; j < 4; ++j) {
        internal[j]=nineth*(-4*point(external,j)
                            +6*(precontrol(external,j)+
				postcontrol(external,j))
                            -2*(point(external,j-1)+point(external,j+1))
                            +3*(precontrol(external,j-1)+
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

  // A constructor for a convex quadrilateral.
  void operator init(triple[] external, triple[] internal=new triple[],
		     triple[] normals=new triple[], pen[] colors=new pen[]) {
    init();
    if(normals.length != 0)
      this.normals=copy(normals);
    if(colors.length != 0)
      this.colors=copy(colors);

    straight=true;

    if(internal.length == 0) {
      internal=new triple[4];
      for(int j=0; j < 4; ++j) {
	internal[j]=nineth*(4*external[j]+2*external[(j+1)%4]+
			    external[(j+2)%4]+2*external[(j+3)%4]);
      }
    }

    triple delta;

    P[0][0]=external[0];
    delta=(external[1]-external[0])/3;
    P[0][1]=external[0]+delta;
    P[1][1]=internal[0];

    P[0][2]=external[1]-delta;
    P[0][3]=external[1];
    delta=(external[2]-external[1])/3;
    P[1][3]=external[1]+delta;
    P[1][2]=internal[1];

    P[2][3]=external[2]-delta;
    P[3][3]=external[2];
    delta=(external[3]-external[2])/3;
    P[3][2]=external[2]+delta;
    P[2][2]=internal[2];

    P[3][1]=external[3]-delta;
    P[3][0]=external[3];
    delta=(external[0]-external[3])/3;
    P[2][0]=external[3]+delta;
    P[2][1]=internal[3];

    P[1][0]=external[0]-delta;
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

  void operator init(triple[][] P, triple[] normals=new triple[],
                     pen[] colors=new pen[]) {
    s=new patch[] {patch(P,normals,colors)};
  }

  void operator init(triple[][][] P, triple[][] normals=new triple[][],
                     pen[][] colors=new pen[][]) {
    s=sequence(new patch(int i) {
        return patch(P[i],normals.length == 0 ? new triple[] : normals[i],
                     colors.length == 0 ? new pen[] : colors[i]);
      },P.length);
  }

  void split(path3 external, triple[] internal=new triple[],
	     triple[] normals=new triple[], pen[] colors=new pen[]) {
    int L=length(external);
    if(L <= 4 || internal.length > 0) {
      s.push(patch(external,internal,normals,colors));
      return;
    }
    if(!cyclic(external)) abort("cyclic path expected");
    real factor=1/L;
    pen[] p;
    triple[] n;
    bool nocolors=colors.length == 0;
    bool nonormals=normals.length == 0;
    triple center;
    for(int i=0; i < L; ++i)
      center += point(external,i);
    center *= factor;
   if(!nocolors) {
     real[] pcenter=rgba(colors[0]);
     for(int i=1; i < L; ++i)
       pcenter += rgba(colors[i]);
     p=new pen[] {rgba(factor*pcenter)};
   }
    if(!nonormals) {
      triple ncenter;
      for(int i=0; i < L; ++i)
	ncenter += normals[i];
      n=new triple[] {factor*ncenter};
    }
    // Use triangles for nonplanar surfaces.
    int step=normal(external) == O ? 1 : 2;
    int i=0;
    int end;
    while((end=i+step) < L) {
      s.push(patch(subpath(external,i,end)--center--cycle,
		   nonormals ? n : concat(normals[i:end+1],n),
		   nocolors ? p : concat(colors[i:end+1],p)));
      i=end;
    }
    s.push(patch(subpath(external,i,L)--center--cycle,
		 nonormals ? n : concat(normals[i:],normals[0:1],n),
		 nocolors ? p : concat(colors[i:],colors[0:1],p)));
  }

  // A constructor for a convex path3.
  void operator init(path3 external, triple[] internal=new triple[],
		     triple[] normals=new triple[], pen[] colors=new pen[]) {
    s=new patch[];
    split(external,internal,normals,colors);
  }

  void operator init(explicit path3[] external,
		     triple[][] internal=new triple[][],
                     triple[][] normals=new triple[][],
		     pen[][] colors=new pen[][]) {
    for(int i=0; i < external.length; ++i)
      split(external[i],
	    internal.length == 0 ? new triple[] : internal[i],
	    normals.length == 0 ? new triple[] : normals[i],
	    colors.length == 0 ? new pen[] : colors[i]);
  }

  void push(path3 external, triple[] internal=new triple[],
            triple[] normals=new triple[] ,pen[] colors=new pen[]) {
    s.push(patch(external,internal,normals,colors));
  }

  // A constructor for a (possibly) nonconvex cyclic path of length <= 4 that
  // returns an array of one or two surfaces in a given plane.
  void operator init (path g, triple plane(pair)=XYplane) {
    int L=length(g);
    if(L > 4 || !cyclic(g))
      abort("cyclic path of length <= 4 expected");
    if(L <= 3) {
      s=new patch[] {patch(path3(g,plane))};
      return;
    }
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

// Construct the surface of rotation generated by rotating g
// from angle1 to angle2 sampled n times about the line c--c+axis.
// An optional surface pen color(int i, real j) may be specified
// to override the color at vertex(i,j).
  void operator init(triple c, path3 g, triple axis, int n=nslice,
		     real angle1=0, real angle2= 360,
		     pen color(int i, real j)=null) {
    axis=unit(axis);
    real w=(angle2-angle1)/n;
    int L=length(g);
    s=new patch[L*n];
    int m=-1;
    transform3[] T=new transform3[n+1];
    transform3 t=rotate(w,c,c+axis);
    T[0]=rotate(angle1,c,c+axis);
    for(int k=1; k <= n; ++k)
      T[k]=T[k-1]*t;

    for(int i=0; i < L; ++i) {
      path3 h=subpath(g,i,i+1);
      path3 r=reverse(h);
      triple max=max(h);
      triple min=min(h);
      triple perp(triple m) {
	static real epsilon=sqrt(realEpsilon);
	triple perp=m-c;
	return perp-dot(perp,axis)*axis;
      }
      triple perp=perp(max);
      real fuzz=epsilon*max(abs(max),abs(min));
      if(abs(perp) < fuzz)
	perp=perp(min);
      perp=unit(perp);
      triple normal=cross(axis,perp);
      triple dir(real j) {return Cos(j)*normal-Sin(j)*perp;}
      real j=angle1;
      transform3 Tk=T[0];
      triple dirj=dir(j);
      for(int k=0; k < n; ++k, j += w) {
        transform3 Tp=T[k+1];
        triple dirp=dir(j+w);
        path3 G=Tk*h{dirj}..{dirp}Tp*r{-dirp}..{-dirj}cycle;
        Tk=Tp;
        dirj=dirp;
        s[++m]=color == null ? patch(G) :
          patch(G,new pen[] {color(i,j),color(i+1,j),color(i+1,j+w),
                             color(i,j+w)});
      }
    }
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
    for(int j=0; j < 4; ++j)
      Si[j]=t*si[j]; 
  }
  
  for(int i=0; i < s.normals.length; ++i)
    S.normals[i]=t*s.normals[i];

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

// Construct a surface from a (possibly) nonconvex planar cyclic path3.
surface planar(path3 p)
{
  if(length(p) <= 3) return surface(patch(p));
  triple n=normal(p);
  if(n == O) return new surface; // p is not planar!
  transform3 T=align(n);
  p=transpose(T)*p;
  return T*shift(0,0,point(p,0).z)*surface(bezulate(path(p)));
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

void draw3D(frame f, patch s, material m, light light=currentlight)
{
  if(!light.on())
    m=emissive(m);
  real granularity=m.granularity >= 0 ? m.granularity : defaultgranularity;
  draw(f,s.P,s.straight,m.p,m.opacity,m.shininess,granularity,
       s.colors.length == 0 ? -s.normal(0.5,0.5) : O,s.colors);
}

void tensorshade(transform t=identity(), frame f, patch s,
                 material m, light light=currentlight, projection P)
{
  tensorshade(f,box(t*s.min(P),t*s.max(P)),m.diffuse(),
              s.colors(m,light),t*project(s.external(),P,1),
              t*project(s.internal(),P));
}

restricted pen[] nullpens={nullpen};
nullpens.cyclic(true);

void draw(transform t=identity(), frame f, surface s, int nu=1, int nv=1,
          material[] surfacepen, pen[] meshpen=nullpens,
	  light light=currentlight, light meshlight=light,
	  projection P=currentprojection)
{
  if(is3D()) {
    for(int i=0; i < s.s.length; ++i) {
      material p=surfacepen[i];
      if(!invisible((pen) p))
	draw3D(f,s.s[i],p,light);
    }
    pen modifiers=thin()+linecap(0);
    for(int k=0; k < s.s.length; ++k) {
      pen meshpen=meshpen[k];
      if(!invisible(meshpen)) {
	meshpen=modifiers+meshpen;
	real step=nu == 0 ? 0 : 1/nu;
	for(int i=0; i <= nu; ++i)
	  draw(f,s.s[k].uequals(i*step),meshpen,meshlight);
	step=nv == 0 ? 0 : 1/nv;
	for(int j=0; j <= nv; ++j)
	  draw(f,s.s[k].vequals(j*step),meshpen,meshlight);
      }
    }
  } else {
    begingroup(f);
    // Sort patches by mean distance from camera
    triple camera=P.camera;
    if(P.infinity) {
      triple m=min(s);
      triple M=max(s);
      camera=P.target+camerafactor*(abs(M-m)+abs(m-P.target))*unit(P.vector());
    }

    real[][] depth;
    
    for(int i=0; i < s.s.length; ++i) {
      real d=abs(camera-s.s[i].cornermean());
      depth.push(new real[] {d,i});
    }

    depth=sort(depth);

    light.T=shiftless(P.modelview());

    // Draw from farthest to nearest
    while(depth.length > 0) {
      real[] a=depth.pop();
      int i=round(a[1]);
      material p=surfacepen[i];
      if(!invisible((pen) p))
        tensorshade(t,f,s.s[i],p,light,P);
      pen meshpen=meshpen[i];
      if(!invisible(meshpen))
        draw(f,t*project(s.s[i].external(),P),meshpen);
    }
    endgroup(f);
  }
}

void draw(transform t=identity(), frame f, surface s, int nu=1, int nv=1,
          material surfacepen=currentpen, pen meshpen=nullpen,
	  light light=currentlight, light meshlight=light,
	  projection P=currentprojection)
{
  material[] surfacepen={surfacepen};
  pen[] meshpen={meshpen};
  surfacepen.cyclic(true);
  meshpen.cyclic(true);
  draw(t,f,s,nu,nv,surfacepen,meshpen,light,meshlight,P);
}

void draw(picture pic=currentpicture, surface s, int nu=1, int nv=1,
          material[] surfacepen, pen[] meshpen=nullpens,
	  light light=currentlight, light meshlight=light)
{
  if(s.empty()) return;

  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      surface S=t*s;
      if(is3D()) {
        draw(f,S,nu,nv,surfacepen,meshpen,light,meshlight);
      } else if(pic != null)
        pic.add(new void(frame f, transform T) {
            draw(T,f,S,nu,nv,surfacepen,meshpen,light,meshlight,P);
          },true);
      if(pic != null) {
        pic.addPoint(min(S,P));
        pic.addPoint(max(S,P));
      }
    },true);
  pic.addPoint(min(s));
  pic.addPoint(max(s));

  pen modifiers;
  if(is3D()) modifiers=thin()+linecap(0);
  for(int k=0; k < s.s.length; ++k) {
    pen meshpen=meshpen[k];
    if(!invisible(meshpen)) {
      meshpen=modifiers+meshpen;
      real step=nu == 0 ? 0 : 1/nu;
      for(int i=0; i <= nu; ++i)
	addPath(pic,s.s[k].uequals(i*step),meshpen);
      step=nv == 0 ? 0 : 1/nv;
      for(int j=0; j <= nv; ++j)
	addPath(pic,s.s[k].vequals(j*step),meshpen);
    }
  }
}

void draw(picture pic=currentpicture, surface s, int nu=1, int nv=1,
          material surfacepen=currentpen, pen meshpen=nullpen,
	  light light=currentlight, light meshlight=light)
{
  material[] surfacepen={surfacepen};
  pen[] meshpen={meshpen};
  surfacepen.cyclic(true);
  meshpen.cyclic(true);
  draw(pic,s,nu,nv,surfacepen,meshpen,light,meshlight);
}

void draw(picture pic=currentpicture, surface s, int nu=1, int nv=1,
          material[] surfacepen, pen meshpen,
	  light light=currentlight, light meshlight=light)
{
  pen[] meshpen={meshpen};
  meshpen.cyclic(true);
  draw(pic,s,nu,nv,surfacepen,meshpen,light,meshlight);
}

surface extrude(path g, triple elongation=Z)
{
  static patch[] allocate;
  path3 G=path3(g);
  path3 G2=shift(elongation)*G;
  return surface(...sequence(new patch(int i) {
        return patch(subpath(G,i,i+1)--subpath(G2,i+1,i)--cycle);
      },length(G)));
}

triple rectify(triple dir) 
{
  real scale=max(abs(dir.x),abs(dir.y),abs(dir.z));
  if(scale != 0) dir *= 0.5/scale;
  dir += (0.5,0.5,0.5);
  return dir;
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
  } else
    fill(f,path(L,project(position,P.t),P),
	 light.color(L.T3*Z,L.p,shiftless(P.modelview())));
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
      if(is3D())
        for(patch S : surface(L,v).s)
          draw3D(f,S,L.p,light);
      if(pic != null)
        fill(project(v,P.t),pic,path(L,P),
	     light.color(L.T3*Z,L.p,shiftless(P.modelview())));
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

restricted patch octant1=patch(X{Z}..{-X}Z..Z{Y}..{-Z}Y{X}..{-Y}cycle,
                               new triple[] {(1,a,a),(a,a^2,1),(a^2,a,1),
                                             (a,1,a)});

restricted surface unithemisphere=surface(octant1,t*octant1,t2*octant1,
					  t3*octant1);
restricted surface unitsphere=surface(octant1,t*octant1,t2*octant1,t3*octant1,
                                      i*octant1,i*t*octant1,i*t2*octant1,
                                      i*t3*octant1);

restricted patch unitfrustum(real t1, real t2)
{
  real s1=interp(t1,t2,1/3);
  real s2=interp(t1,t2,2/3);
  return patch(interp(Z,X,t2)--interp(Z,X,t1){Y}..{-X}interp(Z,Y,t1)--
	       interp(Z,Y,t2){X}..{-Y}cycle,
	       new triple[] {(s2,s2*a,1-s2),(s1,s1*a,1-s1),(s1*a,s1,1-s1),
					  (s2*a,s2,1-s2)});
}

// Return a unitcone constructed from n frusta (the final one being degenerate)
surface unitcone(int n=6)
{
  surface unitcone;
  unitcone.s=new patch[4*n];
  real r=1/3;
  for(int i=0; i < n; ++i) {
    patch s=unitfrustum(i < n-1 ? r^(i+1) : 0,r^i);
    unitcone.s[i]=s;
    unitcone.s[n+i]=t*s;
    unitcone.s[2n+i]=t2*s;
    unitcone.s[3n+i]=t3*s;
  }
  return unitcone;
}

restricted surface unitcone=unitcone();
restricted surface unitsolidcone=surface(patch(unitcircle3)...unitcone.s);

private patch unitcylinder1=patch(X--X+Z{Y}..{-X}Y+Z--Y{X}..{-Y}cycle);

restricted surface unitcylinder=surface(unitcylinder1,t*unitcylinder1,
                                        t2*unitcylinder1,t3*unitcylinder1);

private patch unitplane=patch(new triple[] {O,X,X+Y,Y});
restricted surface unitcube=surface(unitplane,
                                    rotate(90,O,X)*unitplane,
                                    rotate(-90,O,Y)*unitplane,
                                    shift(Z)*unitplane,
                                    rotate(90,X,X+Y)*unitplane,
                                    rotate(-90,Y,X+Y)*unitplane);
restricted surface unitplane=surface(unitplane);
restricted surface unitdisk=surface(unitcircle3);

void dot(frame f, triple v, material p=currentpen,
         light light=nolight, projection P=currentprojection)
{
  pen q=(pen) p;
  if(is3D()) {
    material m=material(p,p.granularity >= 0 ? p.granularity : dotgranularity);
    for(patch s : unitsphere.s)
      draw3D(f,shift(v)*scale3(0.5*dotsize(q))*s,m,light);
  } else dot(f,project(v,P.t),q);
}

void dot(frame f, path3 g, material p=currentpen,
	 projection P=currentprojection)
{
  for(int i=0; i <= length(g); ++i) dot(f,point(g,i),p,P);
}

void dot(frame f, path3[] g, material p=currentpen,
	 projection P=currentprojection)
{
  for(int i=0; i < g.length; ++i) dot(f,g[i],p,P);
}

void dot(picture pic=currentpicture, triple v, material p=currentpen,
         light light=nolight)
{
  pen q=(pen) p;
  real size=dotsize(q);
  pic.add(new void(frame f, transform3 t, picture pic, projection P) {
      if(is3D()) {
	material m=material(p,p.granularity >= 0 ? p.granularity :
			    dotgranularity);
        for(patch s : unitsphere.s)
          draw3D(f,shift(t*v)*scale3(0.5*linewidth(size+q))*s,m,light);
      }
      if(pic != null)
        dot(pic,project(t*v,P.t),q);
    },true);
  triple R=0.5*size*(1,1,1);
  pic.addBox(v,v,-R,R);
}

void dot(picture pic=currentpicture, triple[] v, material p=currentpen)
{
  for(int i=0; i < v.length; ++i) dot(pic,v[i],p);
}

void dot(picture pic=currentpicture, explicit path3 g, material p=currentpen)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p);
}

void dot(picture pic=currentpicture, path3[] g, material p=currentpen)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p);
}

void dot(picture pic=currentpicture, Label L, triple v, align align=NoAlign,
         string format=defaultformat, material p=currentpen)
{
  Label L=L.copy();
  if(L.s == "") {
    if(format == "") format=defaultformat;
    L.s="("+format(format,v.x)+","+format(format,v.y)+","+
      format(format,v.z)+")";
  }
  L.align(align,E);
  L.p((pen) p);
  dot(pic,v,p);
  label(pic,L,v);
}

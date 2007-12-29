import three;
import light;
import graph_settings;

int maxdepth=16;

struct surface {
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

  struct bbox {
    real x,y;
    real X,Y;
    pair min() {return (x,y);}
    pair max() {return (X,Y);}
    void operator init(real x, real y, real X, real Y) {
      this.x=x;
      this.y=y;
      this.X=X;
      this.Y=Y;
    }
    void operator init(pair min, pair max) {
      this.x=min.x;
      this.y=min.y;
      this.X=max.x;
      this.Y=max.y;
    }
  }

  struct bbox3 {
    real x,y,z;
    real X,Y,Z;
    triple min() {return (x,y,z);}
    triple max() {return (X,Y,Z);}
    void operator init(real x, real y, real z, real X, real Y, real Z) {
      this.x=x;
      this.y=y;
      this.z=z;
      this.X=X;
      this.Y=Y;
      this.Z=Z;
    }
    void operator init(triple min, triple max) {
      this.x=min.x;
      this.y=min.y;
      this.z=min.z;
      this.X=max.x;
      this.Y=max.y;
      this.Z=max.z;
    }
  }

  struct bboxes {
    real[] x,y;
    real[] X,Y;
    void operator init(int n) {
      x=new real[n];
      y=new real[n];
      X=new real[n];
      Y=new real[n];
    }
    void set(int i,bbox b) {
      x[i]=b.x;
      X[i]=b.X;
      y[i]=b.y;
      Y[i]=b.Y;
    }
  }

  void write(bbox a) {
    write(a.min(),a.max());
  }

  struct bboxes3 {
    real[] x,y,z;
    real[] X,Y,Z;

    void operator init(int n) {
      x=new real[n];
      y=new real[n];
      z=new real[n];
      X=new real[n];
      Y=new real[n];
      Z=new real[n];
    }
    void set(int i,bbox3 b) {
      x[i]=b.x;
      X[i]=b.X;
      y[i]=b.y;
      Y[i]=b.Y;
      z[i]=b.z;
      Z[i]=b.Z;
    }
  }

  void write(bbox3 a) {
    write(a.min(),a.max());
  }

  private int minindex(real[] x) {return find(x == minbound(x));}
  private int maxindex(real[] x) {return find(x == maxbound(x));}

  private bbox3 bbox3(... triple[] points) {
    return bbox3(minbound(points),maxbound(points));
  }

  private bbox bbox(... pair[] points) {
    return bbox(minbound(points),maxbound(points));
  }

  private triple[] RtrnPointOnSrfc(triple a, triple cp1, triple cp2, triple b) {
    triple m0=0.5*(a+cp1);
    triple m1=0.5*(cp1+cp2);
    triple m2=0.5*(cp2+b);
    triple m3=0.5*(m0+m1);
    triple m4=0.5*(m1+m2);
    triple m5=0.5*(m3+m4);

    return new triple[] {m0,m3,m5,m4,m2};
  }

  //Splits a Bezier surface into 4 subsurfaces.
  private triple[][] splitsurface4(triple[] p) {
    // Find new control points.
    triple[] c0=RtrnPointOnSrfc(p[0],p[1],p[2],p[3]);
    triple[] c1=RtrnPointOnSrfc(p[4],p[5],p[6],p[7]);
    triple[] c2=RtrnPointOnSrfc(p[8],p[9],p[10],p[11]);
    triple[] c3=RtrnPointOnSrfc(p[12],p[13],p[14],p[15]);

    triple[] c4=RtrnPointOnSrfc(p[12],p[8],p[4],p[0]);
    triple[] c5=RtrnPointOnSrfc(c3[0],c2[0],c1[0],c0[0]);
    triple[] c6=RtrnPointOnSrfc(c3[1],c2[1],c1[1],c0[1]);
    triple[] c7=RtrnPointOnSrfc(c3[2],c2[2],c1[2],c0[2]);

    triple[] c8=RtrnPointOnSrfc(c3[3],c2[3],c1[3],c0[3]);
    triple[] c9=RtrnPointOnSrfc(c3[4],c2[4],c1[4],c0[4]);
    triple[] c10=RtrnPointOnSrfc(p[15],p[11],p[7],p[3]);

    //Set up 4 Bezier subsurfaces.
    triple[] bss0={c4[2],c5[2],c6[2],c7[2],c4[1],c5[1],c6[1],c7[1],
		   c4[0],c5[0],c6[0],c7[0],p[12],c3[0],c3[1],c3[2]};
    triple[] bss1={p[0],c0[0],c0[1],c0[2],c4[4],c5[4],c6[4],c7[4],
		   c4[3],c5[3],c6[3],c7[3],c4[2],c5[2],c6[2],c7[2]};
    triple[] bss2={c0[2],c0[3],c0[4],p[3],c7[4],c8[4],c9[4],c10[4],
		   c7[3],c8[3],c9[3],c10[3],c7[2],c8[2],c9[2],c10[2]};
    triple[] bss3={c7[2],c8[2],c9[2],c10[2],c7[1],c8[1],c9[1],c10[1],
		   c7[0],c8[0],c9[0],c10[0],c3[2],c3[3],c3[4],p[15]};

    return new triple[][] {bss0,bss1,bss2,bss3};
  }

  // Check if an extremum is on the surface, in which case it is the extremum
  // we want.
  private bbox ipointonsrfc(pair[] p) {
    return bbox(p[0],p[3],p[12],p[15]);
  }

  private bbox3 ipointonsrfc(triple[] p) {
    return bbox3(p[0],p[3],p[12],p[15]);
  }

  private bbox ipointonsrfc(triple[] p, projection P) {
    return bbox(project(p[0],P),project(p[3],P),project(p[12],P),
		project(p[15],P));
  }

  // If both the min and max values of the bbox are points on the surface, 
  // the patch is explored and we should cease examining it.
  private bool isexplored(triple[] p, pair m, pair M, projection P) {
    bbox c=ipointonsrfc(p,P);
    return m == c.min() && M == c.max();
  }

  private bool isexplored(triple[] p, triple m, triple M) {
    bbox3 c=ipointonsrfc(p);
    return m == c.min() && M == c.max();
  }

  private bboxes projboxes(triple[][] ss, projection P) {
    bboxes bboxes=bboxes(ss.length);
    for(int i=0; i < ss.length; ++i) {
      triple[] ssi=ss[i];
      bboxes.set(i,bbox(...sequence(new pair(int j) {
	      return project(ssi[j],P);
	    },16)));
    } 
    return bboxes;
  }

  private int[] removal(triple[][] ss, bboxes b, real xlb, real xsb, real ylb, 
			real ysb, bool[] found, projection P) {
    // Remove areas of the surface that should not be explored.
    int[] toremove;
    for(int i=0; i < ss.length; ++i) {
      if((found[0] || b.X[i] <= xlb) && (found[1] || b.x[i] >= xsb) && 
	 (found[2] || b.Y[i] <= ylb) && (found[3] || b.y[i] >= ysb))
	toremove.push(i);
      else if(isexplored(ss[i],(b.x[i],b.y[i]),(b.X[i],b.Y[i]),P))
	toremove.push(i);
    }
    return toremove;
  }

  private int[] removal(triple[][] ss, bboxes3 b, real xlb, real xsb, 
			real ylb, real ysb, real zlb, real zsb, bool[] found) {
    int[] toremove;
    for(int i=0; i < ss.length; ++i) {
      if((found[0] || b.X[i] <= xlb) && (found[1] || b.x[i] >= xsb) && 
	 (found[2] || b.Y[i] <= ylb) && (found[3] || b.y[i] >= ysb) &&
	 (found[4] || b.Z[i] <= zlb) && (found[5] || b.z[i] >= zsb))
	toremove.push(i);
      else if(isexplored(ss[i],(b.x[i],b.y[i],b.z[i]),(b.X[i],b.Y[i],b.Z[i]))) 
	toremove.push(i);
    }
    return toremove;
  }

  private void iteration(triple[][] sfcs, real[] rvalues, bool[] found,
			 projection P) {
    static int depth=0;

    // Refine current partitioning.
    triple[][] ss=splitsurface4(sfcs[0]);
    for(int i=1; i < sfcs.length; ++i)
      ss.append(splitsurface4(sfcs[i]));

    bboxes bxs=projboxes(ss,P);

    // See if an extremum has been attained.
    real xlb,xsb,ylb,ysb;

    bool stop=depth >= maxdepth;

    if(!found[0]) {
      int xlarge=maxindex(bxs.X);
      xlb=bxs.x[xlarge];
      if(ipointonsrfc(ss[xlarge],P).X == bxs.X[xlarge] || stop) {
	found[0]=true;
	rvalues[0]=bxs.X[xlarge];
      }
    }
    if(!found[1]) {
      int xsmall=minindex(bxs.x);
      xsb=bxs.X[xsmall];
      if(ipointonsrfc(ss[xsmall],P).x == bxs.x[xsmall] || stop) {
	found[1]=true;
	rvalues[1]=bxs.x[xsmall];
      }
    }
    if(!found[2]) {
      int ylarge=maxindex(bxs.Y);
      ylb=bxs.y[ylarge];
      if(ipointonsrfc(ss[ylarge],P).Y == bxs.Y[ylarge] || stop) {
	found[2]=true;
	rvalues[2]=bxs.Y[ylarge];
      }
    }
    if(!found[3]) {
      int ysmall=minindex(bxs.y);
      ysb=bxs.Y[ysmall];
      if(ipointonsrfc(ss[ysmall],P).y == bxs.y[ysmall] || stop) {
	found[3]=true;
	rvalues[3]=bxs.y[ysmall];
      }
    }

    // Stopping condition.
    if(all(found)) return;

    int[] toremove=removal(ss,bxs,xlb,xsb,ylb,ysb,found,P);
    ss=ss[complement(toremove,ss.length)];
  
    ++depth;
    iteration(ss,rvalues,found,P);
    --depth;
  }

  // Finds the projection bounding box of a given surface.
  bbox bbox(triple[][] pts, projection P=currentprojection) {
    triple[] points=new triple[16];
    int k=0;
    for(int i=0; i < 4; ++i) {
      triple[] ptsi=pts[i];
      for(int j=0; j < 4; ++j) {
	points[k]=ptsi[j];
	++k;
      }
    }

    // Splits surface.
    triple[][] ss=splitsurface4(points);

    bboxes bxs=projboxes(ss,P);

    // Checks if extrema have been attained.
    int xsmall=minindex(bxs.x);
    int ysmall=minindex(bxs.y);
    int xlarge=maxindex(bxs.X);
    int ylarge=maxindex(bxs.Y);

    real xlb=bxs.x[xlarge];
    real ylb=bxs.y[ylarge];
    real xsb=bxs.X[xsmall];
    real ysb=bxs.Y[ysmall];

    real[] rvalues=new real[4];
    bool[] found=new bool[4];
    for(int i=0; i < found.length; ++i)
      found[i]=false;

    if(ipointonsrfc(ss[xlarge],P).X == bxs.X[xlarge]) {
      found[0]=true;
      rvalues[0]=bxs.X[xlarge];
    }
    if(ipointonsrfc(ss[xsmall],P).x == bxs.x[xsmall]) {
      found[1]=true;
      rvalues[1]=bxs.x[xsmall];
    }
    if(ipointonsrfc(ss[ylarge],P).Y == bxs.Y[ylarge]) {
      found[2]=true;
      rvalues[2]=bxs.Y[ylarge];
    }
    if(ipointonsrfc(ss[ysmall],P).y == bxs.y[ysmall]) {
      found[3]=true;
      rvalues[3]=bxs.y[ysmall];
    }

    if(!all(found)) {
      int[] toremove=removal(ss,bxs,xlb,xsb,ylb,ysb,found,P);
      ss=ss[complement(toremove,ss.length)];

      iteration(ss,rvalues,found,P);
    }

    // Prepare return value and return it.
    return bbox(rvalues[1],rvalues[3],rvalues[0],rvalues[2]);
  }


  private void bbox3it(triple[][] sfcs, real[] rvalues, bool[] found) {
    // Refine current partitioning.
    triple[][] ss=splitsurface4(sfcs[0]);
    for(int i=1; i < sfcs.length; ++i)
      ss.append(splitsurface4(sfcs[i]));

    bboxes3 bxs=bboxes3(ss.length);
    for(int i=0; i < ss.length; ++i)
      bxs.set(i,bbox3(...ss[i]));

    // See if an extremum has been attained.
    real xlb,xsb,ylb,ysb,zlb,zsb;
    if(!found[0]) {
      int xlarge=maxindex(bxs.X);
      xlb=bxs.x[xlarge];
      if(ipointonsrfc(ss[xlarge]).X == bxs.X[xlarge]) {
	found[0]=true;
	rvalues[0]=bxs.X[xlarge];
      }
    }
    if(!found[1]) {
      int xsmall=minindex(bxs.x);
      xsb=bxs.X[xsmall];
      if(ipointonsrfc(ss[xsmall]).x == bxs.x[xsmall]) {
	found[1]=true;
	rvalues[1]=bxs.x[xsmall];
      }
    }
    if(!found[2]) {
      int ylarge=maxindex(bxs.Y);
      ylb=bxs.y[ylarge];
      if(ipointonsrfc(ss[ylarge]).Y == bxs.Y[ylarge]) {
	found[2]=true;
	rvalues[2]=bxs.Y[ylarge];
      }
    }
    if(!found[3]) {
      int ysmall=minindex(bxs.y);
      ysb=bxs.Y[ysmall];
      if(ipointonsrfc(ss[ysmall]).y == bxs.y[ysmall]) {
	found[3]=true;
	rvalues[3]=bxs.y[ysmall];
      }
    }
    if(!found[4]) {
      int zlarge=maxindex(bxs.Z);
      zlb=bxs.z[zlarge];
      if(ipointonsrfc(ss[zlarge]).Z == bxs.Z[zlarge]) {
	found[4]=true;
	rvalues[4]=bxs.Z[zlarge];
      }
    }
    if(!found[5]) {
      int zsmall=minindex(bxs.z);
      zsb=bxs.Z[zsmall];
      if(ipointonsrfc(ss[zsmall]).z == bxs.z[zsmall]) {
	found[5]=true;
	rvalues[5]=bxs.z[zsmall];
      }
    }

    // Stopping conditions.
    if(all(found)) return;

    int[] toremove=removal(ss,bxs,xlb,xsb,ylb,ysb,zlb,zsb,found);
    ss=ss[complement(toremove,ss.length)];

    bbox3it(ss,rvalues,found);
  }

  bbox3 bbox3(triple[][] pts) {
    triple[] points=new triple[16];
    int k=0;
    for(int i=0; i < 4; ++i) {
      triple[] ptsi=pts[i];
      for(int j=0; j < 4; ++j) {
	points[k]=ptsi[j];
	++k;
      }
    }

    // Split surface.
  
    triple[][] split4=splitsurface4(points);
    triple[][] split16=splitsurface4(split4[0]);
    for(int i=1; i < split4.length; ++i)
      split16.append(splitsurface4(split4[i]));
    triple[][] ss=splitsurface4(split16[0]);
    for(int i=1; i < split16.length; ++i)
      ss.append(splitsurface4(split16[i]));

    bboxes3 bxs=bboxes3(ss.length);
    for(int i=0; i < ss.length; ++i)
      bxs.set(i,bbox3(...ss[i]));

    // Check if extrema have been attained.
    int xsmall=minindex(bxs.x);
    int ysmall=minindex(bxs.y);
    int zsmall=minindex(bxs.z);
    int xlarge=maxindex(bxs.X);
    int ylarge=maxindex(bxs.Y);
    int zlarge=maxindex(bxs.Z);

    real xlb=bxs.x[xlarge];
    real ylb=bxs.y[ylarge];
    real zlb=bxs.z[zlarge];
    real xsb=bxs.X[xsmall];
    real ysb=bxs.Y[ysmall];
    real zsb=bxs.Z[zsmall];

    real[] rvalues=new real[6];
    bool[] found=new bool[6];
    for(int i=0; i < found.length; ++i)
      found[i]=false;

    if(ipointonsrfc(ss[xlarge]).X == bxs.X[xlarge]) {
      found[0]=true;
      rvalues[0]=bxs.X[xlarge];
    }
    if(ipointonsrfc(ss[xsmall]).x == bxs.x[xsmall]) {
      found[1]=true;
      rvalues[1]=bxs.x[xsmall];
    }
    if(ipointonsrfc(ss[ylarge]).Y == bxs.Y[ylarge]) {
      found[2]=true;
      rvalues[2]=bxs.Y[ylarge];
    }
    if(ipointonsrfc(ss[ysmall]).y == bxs.y[ysmall]) {
      found[3]=true;
      rvalues[3]=bxs.y[ysmall];
    }

    if(ipointonsrfc(ss[zlarge]).Z == bxs.Z[zlarge]) {
      found[4]=true;
      rvalues[4]=bxs.Z[zlarge];
    }
    if(ipointonsrfc(ss[zsmall]).z == bxs.z[zsmall]) {
      found[5]=true;
      rvalues[5]=bxs.z[zsmall];
    }

    if(!all(found)) {
      int[] toremove=removal(ss,bxs,xlb,xsb,ylb,ysb,zlb,zsb,found);
      ss=ss[complement(toremove,ss.length)];

      if(ss.length > 0)
	bbox3it(ss,rvalues,found);
    }

    // Prepare return value and return it.
    return bbox3(rvalues[1],rvalues[3],rvalues[5],
		 rvalues[0],rvalues[2],rvalues[4]);
  }

  struct bounds {
    bool empty=true;
    projection Q;
    bbox b;
    void init(projection Q) {
      if(empty || Q != this.Q) {
	b=bbox(P,Q);
	this.Q=Q;
	empty=false;
      }
    }
    pair min(projection P) {init(P); return b.min();}
    pair max(projection P) {init(P); return b.max();}
  }

  struct bounds3 {
    bool empty=true;
    bbox3 b;
    void init() {
      if(empty) {
	b=bbox3(P);
	empty=false;
      }
    }
    triple min() {init(); return b.min();}
    triple max() {init(); return b.max();}
  }

  bounds bounds;
  bounds3 bounds3;

  pair min(projection P) {return bounds.min(P);}
  pair max(projection P) {return bounds.max(P);}

  triple min() {return bounds3.min();}
  triple max() {return bounds3.max();}

  void init(triple[][] P) {
    bounds.empty=true;
    bounds3.empty=true;
    this.P=copy(P);
  }

  void init(path3 external, triple[] internal=new triple[]) {
    bounds.empty=true;
    bounds3.empty=true;
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
 
surface operator cast(triple[][] P)
{
  surface s;
  s.init(P);
  return s;
}

surface surface(triple[][] P)
{
  surface s;
  s.init(P);
  return s;
}

path3[] bbox3(surface s)
{
  return box(s.min(),s.max());
}

surface surface(path3 external, triple[] internal=new triple[]) 
{
  surface s;
  s.init(external,internal);
  return s;
}

triple min(surface s) {return s.min();}
triple max(surface s) {return s.max();}

pair min(surface s, projection P) {return s.min(P);}
pair max(surface s, projection P) {return s.max(P);}

surface subsurfaceu(surface s, real ua, real ub)
{
  path3 G=s.uequals(ua)&subpath(s.vequals(1),ua,ub)&
    reverse(s.uequals(ub));
  path3 w=subpath(s.vequals(0),ub,ua);
  path3 i1=s.P[0][1]..controls s.P[1][1] and s.P[2][1]..s.P[3][1];
  path3 i2=s.P[0][2]..controls s.P[1][2] and s.P[2][2]..s.P[3][2];
  path3 s1=subpath(i1,ua,ub);
  path3 s2=subpath(i2,ua,ub);
  return surface(G..controls postcontrol(w,0) and precontrol(w,1)..cycle,
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

  return surface(G..controls precontrol(w,1) and postcontrol(w,0)..cycle,
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
                 projection P=currentprojection, int ninterpolate=1)
{
  path[] b=box(min(s,P),max(s,P));
  tensorshade(pic,box(min(b),max(b)),surfacepen,s.colors(surfacepen,light),
              project(s.external(),P,1),project(s.internal(),P));
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
        real d=abs(camera-0.25*(point(s0,j/nv)+point(s0,(j+1)/nv)+
                                point(s1,j/nv)+point(s1,(j+1)/nv)));
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

// Contour routines written by Radoslav Marinov and John Bowman.
	 
import graph_settings;

real eps=100*realEpsilon;

//                         1  
//             6 +-------------------+ 5
//               | \               / |
//               |   \          /    |
//               |     \       /     |
//               |       \   /       |
//             2 |         X         | 0
//               |       /   \       |
//               |     /       \     |
//               |   /           \   |
//               | /               \ |
//             7 +-------------------+ 4 or 8
//                         3  

private struct segment
{
  bool active;
  pair a,b;	  // Endpoints; a is always an edge point if one exists.
  int c;           // Contour value.
  int edge;        // -1: interior, 0 to 3: edge,
                          // 4-8: single edge vertex, 9: double edge vertex.
}

segment operator init() {return new segment;}

// Case 1: line passes through two vertices of a triangle
private segment case1(pair p0, pair p1, int edge)
{
  // Will cause a duplicate guide; luckily case1 is rare
  segment rtrn;
  rtrn.active=true;
  rtrn.a=p0;
  rtrn.b=p1;
  rtrn.edge=edge;
  return rtrn;
}

// Case 2: line passes through a vertex and a side of a triangle
// (the first vertex passed and the side between the other two)
private segment case2(pair p0, pair p1, pair p2,
		      real v0, real v1, real v2, int edge)
{
  segment rtrn;
  pair val=interp(p1,p2,abs(v1/(v2-v1)));
  rtrn.active=true;
  if(edge < 4) {
    rtrn.a=val;
    rtrn.b=p0;
  } else {
    rtrn.a=p0;
    rtrn.b=val;
  }
  rtrn.edge=edge;
  return rtrn;
}

// Case 3: line passes through two sides of a triangle
// (through the sides formed by the first & second, and second & third
// vertices)
private segment case3(pair p0, pair p1, pair p2,
		      real v0, real v1, real v2, int edge=-1)
{
  segment rtrn;
  rtrn.active=true;
  rtrn.a=interp(p1,p0,abs(v1/(v0-v1)));
  rtrn.b=interp(p1,p2,abs(v1/(v2-v1)));
  rtrn.edge=edge;
  return rtrn;
}

// Check if a line passes through a triangle, and draw the required line.
private segment checktriangle(pair p0, pair p1, pair p2,
			      real v0, real v1, real v2, int edge)
{
  // default null return  
  static segment dflt;

  real eps=eps*max(abs(v0),abs(v1),abs(v2));
  
  if(v0 < -eps) {
    if(v1 < -eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return dflt; // nothing to do
      else return case3(p0,p2,p1,v0,v2,v1);
    } else if(v1 <= eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return case1(p1,p2,5+edge);
      else return case2(p1,p0,p2,v1,v0,v2,5+edge);
    } else {
      if(v2 < -eps) return case3(p0,p1,p2,v0,v1,v2,edge);
      else if(v2 <= eps) 
	return case2(p2,p0,p1,v2,v0,v1,edge);
      else return case3(p1,p0,p2,v1,v0,v2,edge);
    } 
  } else if(v0 <= eps) {
    if(v1 < -eps) {
      if(v2 < -eps) return dflt; // nothing to do
      else if(v2 <= eps) return case1(p0,p2,4+edge);
      else return case2(p0,p1,p2,v0,v1,v2,4+edge);
    } else if(v1 <= eps) {
      if(v2 < -eps) return case1(p0,p1,9);
      else if(v2 <= eps) return dflt; // use finer partitioning.
      else return case1(p0,p1,9);
    } else {
      if(v2 < -eps) return case2(p0,p1,p2,v0,v1,v2,4+edge);
      else if(v2 <= eps) return case1(p0,p2,4+edge);
      else return dflt; // nothing to do
    } 
  } else {
    if(v1 < -eps) {
      if(v2 < -eps) return case3(p1,p0,p2,v1,v0,v2,edge);
      else if(v2 <= eps)
	return case2(p2,p0,p1,v2,v0,v1,edge);
      else return case3(p0,p1,p2,v0,v1,v2,edge);
    } else if(v1 <= eps) {
      if(v2 < -eps) return case2(p1,p0,p2,v1,v0,v2,5+edge);
      else if(v2 <= eps) return case1(p1,p2,5+edge);
      else return dflt; // nothing to do
    } else {
      if(v2 < -eps) return case3(p0,p2,p1,v0,v2,v1);
      else if(v2 <= eps) return dflt; // nothing to do
      else return dflt; // nothing to do
    } 
  }      
}

typedef guide interpolate(... guide[]);

// return contour guides for a 2D data array, using a triangle mesh
// f:        two-dimensional array of real data values
// a,b:      lower-left and upper-right vertices of contour domain
// c:        array of contour values
// join:     interpolation operator (e.g. operator -- or operator ..)
guide[][] contour(real[][] f, real[][] midpoint=new real[][],
		  pair a, pair b, real[] c,
		  interpolate join=operator --)
{
  int nx=f.length-1;
  int ny=nx > 0 ? f[0].length-1 : 0;
  
  c=sort(c);
  bool midpoints=midpoint.length > 0;
  
  // check if boundaries are good
  if(b.x <= a.x || b.y <= a.y) {
    abort("bad contour domain: the second vertex (b) must be above and to the right of the first one (a).");
    
  } 

  segment segments[][][]=new segment[nx][ny][0];

  real dx=(b.x-a.x)/nx;
  real dy=(b.y-a.y)/ny;
  
  // go over region a rectangle at a time
  for(int i=0; i < nx; ++i) {
    real x=a.x+i*dx;
    real[] fi=f[i];
    real[] fi1=f[i+1];
    segment[][] segmentsi=segments[i];
    for(int j=0; j < ny; ++j) {
      real y=a.y+j*dy;
      segment[] segmentsij=segmentsi[j];
      
      // define points
      pair bleft=(x,y);
      pair bright=(x+dx,y);
      pair tleft=(x,y+dy);
      pair tright=(x+dx,y+dy);
      pair middle=0.5*(bleft+tright);
   
      real f00=fi[j];
      real f01=fi[j+1];
      real f10=fi1[j];
      real f11=fi1[j+1];
      
      int checkcell(int cnt) {
	real C=c[cnt];
        real vertdat0=f00-C;  // lower-left vertex
        real vertdat1=f10-C;  // lower-right vertex
        real vertdat2=f01-C;  // upper-left vertex
        real vertdat3=f11-C;  // upper-right vertex

        // optimization: we make sure we don't work with empty rectangles
        int countm=0;
        int countz=0;
        int countp=0;
	
	void check(real vertdat) {
	  if(vertdat < -eps) ++countm;
          else {
	    if(vertdat <= eps) ++countz; 
	    else ++countp;
	  }
	}
	
	check(vertdat0);
	check(vertdat1);
	check(vertdat2);
	check(vertdat3);

        if(countm == 4) return 1;  // nothing to do 
        if(countp == 4) return -1; // nothing to do 
        if((countm == 3 || countp == 3) && countz == 1) return 0;

        // evaluate point at middle of rectangle (to set up triangles)
	real vertdat4=midpoints ? midpoint[i][j]-C :
	  0.25*(vertdat0+vertdat1+vertdat2+vertdat3);
      
        // go through the triangles
	
	void addseg(segment seg) { 
	  if(seg.active) {
	    seg.c=cnt;
	    segmentsij.push(seg);
	  }
	}

	addseg(checktriangle(bright,tright,middle,
			     vertdat1,vertdat3,vertdat4,0));
	addseg(checktriangle(tright,tleft,middle,
			     vertdat3,vertdat2,vertdat4,1));
	addseg(checktriangle(tleft,bleft,middle,
			     vertdat2,vertdat0,vertdat4,2));
	addseg(checktriangle(bleft,bright,middle,
			     vertdat0,vertdat1,vertdat4,3));
	return 0;
      }
      
      void process(int l, int u) {
	if(l >= u) return;
	int i=quotient(l+u,2);
	int sign=checkcell(i);
	if(sign == -1) process(i+1,u);
	else if(sign == 1) process(l,i);
	else {
	  process(l,i);
	  process(i+1,u);
	}
      }
    
      process(0,c.length);
    }
  }


  // set up return value
  pair[][][] points=new pair[c.length][0][0];

  for(int i=0; i < nx; ++i) {
    segment[][] segmentsi=segments[i];
    for(int j=0; j < ny; ++j) {
      segment[] segmentsij=segmentsi[j];
      for(int k=0; k < segmentsij.length; ++k) {
	segment C=segmentsij[k];

	if(!C.active) continue;

	pair[] g=new pair[] {C.a,C.b};
	segmentsij[k].active=false;

	int forward(int I, int J, bool first=true) {
	  if(I >= 0 && I < nx && J >= 0 && J < ny) {
	    segment[] segmentsIJ=segments[I][J];
	    for(int l=0; l < segmentsIJ.length; ++l) {
	      segment D=segmentsIJ[l];
	      if(!D.active) continue;
	      if(abs(D.a-g[g.length-1]) < eps) {
		g.push(D.b);
		segmentsIJ[l].active=false;
		if(D.edge >= 0 && !first) return D.edge;
		first=false;
		l=-1;
	      } else if(abs(D.b-g[g.length-1]) < eps) {
		g.push(D.a);
		segmentsIJ[l].active=false;
		if(D.edge >= 0 && !first) return D.edge;
		first=false;
		l=-1;
	      }
	    }
	  }
	  return -1;
	}
	
	int backward(int I, int J, bool first=true) {
	  if(I >= 0 && I < nx && J >= 0 && J < ny) {
	    segment[] segmentsIJ=segments[I][J];
	    for(int l=0; l < segmentsIJ.length; ++l) {
	      segment D=segmentsIJ[l];
	      if(!D.active) continue;
	      if(abs(D.a-g[0]) < eps) {
		g.insert(0,D.b);
		segmentsIJ[l].active=false;
		if(D.edge >= 0 && !first) return D.edge;
		first=false;
		l=-1;
	      } else if(abs(D.b-g[0]) < eps) {
		g.insert(0,D.a);
		segmentsIJ[l].active=false;
		if(D.edge >= 0 && !first) return D.edge;
		first=false;
		l=-1;
	      }
	    }
	  }
	  return -1;
	}
	
	void follow(int f(int, int, bool first=true), int edge) {
	  int I=i;
	  int J=j;
	  while(true) {
	    static int ix[]={1,0,-1,0};
	    static int iy[]={0,1,0,-1};
	    if(edge >= 0 && edge < 4) {
	      I += ix[edge];
	      J += iy[edge];
	      edge=f(I,J);
	    } else {
	      if(edge == -1) break;
	      if(edge < 9) {
		int edge0=(edge-5) % 4;
		int edge1=(edge-4) % 4;
		int ix0=ix[edge0];
		int iy0=iy[edge0];
		I += ix0;
		J += iy0;
		// Search all 3 corner cells
		if((edge=f(I,J)) == -1) {
		  I += ix[edge1];
		  J += iy[edge1];
		  if((edge=f(I,J)) == -1) {
		    I -= ix0;
		    J -= iy0;
		    edge=f(I,J);
		  }
		}
	      } else {
		// Two edge vertices: search all 8 surrounding cells
		int I0=I;
		int J0=J;
		for(int i=-1; i <= 1; ++i) {
		  for(int j=-1; j <= 1; ++j) {
		    if(i == 0 && j == 0) continue;
		    I=I0+i;
		    J=J0+j;
		    if((edge=f(I,J)) >= 0) break;
		  }
		}
	      }
	    }
	  }
	}

	// Follow contour in cell
	int edge=forward(i,j,first=false);

	// Follow contour forward outside of cell
	follow(forward,edge);

	// Follow contour backward outside of cell
	follow(backward,C.edge);

	points[C.c].push(g);
      }      
    }
  }

  guide[][] result=new guide[c.length][0];
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] pointscnt=points[cnt];
    guide[] resultcnt=result[cnt]=new guide[pointscnt.length];
    for(int i=0; i < pointscnt.length; ++i) {
      pair[] pts=pointscnt[i];
      guide gd=pts[0];
      for(int j=1; j < pts.length-1; ++j)
      	gd=join(gd,pts[j]);
      if(abs(pts[0]-pts[pts.length-1]) < eps)
        gd=gd..cycle;
      else
	gd=join(gd,pts[pts.length-1]);
      resultcnt[i]=gd;
    }
  }

  return result;
}

// return contour guides for a real-valued function
// f:        real-valued function of two real variables
// a,b:      lower-left and upper-right vertices of contour domain
// c:        array of contour values
// nx,ny:    subdivisions on x and y axes (affects accuracy)
// join:     interpolation operator (e.g. operator -- or operator ..)
guide[][] contour(real f(real, real), pair a, pair b,
		  real[] c, int nx=ngraph, int ny=nx,
		  interpolate join=operator --)
{
  // evaluate function at points and midpoints
  real[][] dat=new real[nx+1][ny+1];
  real[][] midpoint=new real[nx+1][ny+1];
  
  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i/nx);
    real x2=interp(a.x,b.x,(i+0.5)/nx);
    real[] dati=dat[i];
    real[] midpointi=midpoint[i];
    for(int j=0; j <= ny; ++j) {
      dati[j]=f(x,interp(a.y,b.y,j/ny));
      midpointi[j]=f(x2,interp(a.y,b.y,(j+0.5)/ny));
    }
  }

  return contour(dat,midpoint,a,b,c,join);
}
  
void draw(picture pic=currentpicture, Label[] L=new Label[],
	  guide[][] g, pen[] p)
{
  begingroup(pic);
  for(int cnt=0; cnt < g.length; ++cnt) {
    for(int i=0; i < g[cnt].length; ++i)
      draw(pic,g[cnt][i],p[cnt]);
  }
  if(L.length > 0) {
    for(int cnt=0; cnt < g.length; ++cnt) {
      for(int i=0; i < g[cnt].length; ++i) {
	Label Lcnt=L[cnt];
	if(Lcnt.s != "" && size(g[cnt][i]) > 1)
	  label(pic,Lcnt,g[cnt][i],p[cnt]);
      }
    }
  }
  endgroup(pic);
}

void draw(picture pic=currentpicture, Label[] L=new Label[],
	  guide[][] g, pen p=currentpen)
{
  draw(pic,L,g,sequence(new pen(int) {return p;},g.length));
}

// non-regularly spaced points routines:

// check existing guides and adds new segment to them if possible,
// or otherwise store segment as a new guide
private void addseg(pair[][] gds, segment seg)
{ 
  if(!seg.active) return;
  // search for a path to extend
  for(int i=0; i < gds.length; ++i) {
    pair[] gd=gds[i];
    if(abs(gd[0]-seg.b) < eps) {
      gd.insert(0,seg.a);
      return;
    } else if(abs(gd[gd.length-1]-seg.b) < eps) {
      gd.push(seg.a); 
      return;
    } else if(abs(gd[0]-seg.a) < eps) {
      gd.insert(0,seg.b);
      return;
    } else if(abs(gd[gd.length-1]-seg.a) < eps) {  
      gd.push(seg.b);
      return;
    }
  }
 
  // in case nothing is found
  pair[] segm;
  segm=new pair[] {seg.a,seg.b}; 
  gds.push(segm);
  
  return;
}

guide[][] contour(pair[] points, real[] values, 
		  real[] c, interpolate join=operator --)
{
  int[][] trn=triangulate(points);

  // array to store guides found so far
  pair[][][] gds=new pair[c.length][0][0];
	
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] gdscnt=gds[cnt];
    for(int i=0; i < trn.length; ++i) {
      int i0=trn[i][0], i1=trn[i][1], i2=trn[i][2];
      addseg(gdscnt,checktriangle(points[i0],points[i1],points[i2],
				  values[i0]-c[cnt],values[i1]-c[cnt],
				  values[i2]-c[cnt],0));
    }
  }

  // connect existing paths
  // use to reverse an array, omitting the first point
  int[] reverseF(int n) {return sequence(new int(int x){return n-1-x;},n-1);}
  // use to reverse an array, omitting the last point
  int[] reverseL(int n) {return sequence(new int(int x){return n-2-x;},n-1);}
  
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] gdscnt=gds[cnt];
    for(int i=0; i < gdscnt.length; ++i) {
      pair[] gig=gdscnt[i];
      int Li=gig.length;
      for(int j=i+1; j < gdscnt.length; ++j) {
        pair[] gjg=gdscnt[j];
	int Lj=gjg.length;
        if(abs(gig[0]-gjg[0]) < eps) { 
	  gdscnt[j]=gjg[reverseF(Lj)];
	  gdscnt[j].append(gig);
          gdscnt.delete(i); 
          --i; 
          break;
        } else if(abs(gig[0]-gjg[Lj-1]) < eps) {
	  gig.delete(0);
	  gdscnt[j].append(gig);
          gdscnt.delete(i);
          --i;
          break;
        } else if(abs(gig[Li-1]-gjg[0]) < eps) {
	  gjg.delete(0);
	  gig.append(gjg);
	  gdscnt[j]=gig;
          gdscnt.delete(i);
          --i;
          break;
        } else if(abs(gig[Li-1]-gjg[Lj-1]) < eps) {
	  gig.append(gjg[reverseL(Lj)]);
          gdscnt[j]=gig;
          gdscnt.delete(i);
          --i;
          break;
        } 
      }
    }
  }

  // set up return value
  guide[][] result=new guide[c.length][0];
  for(int cnt=0; cnt < c.length; ++cnt) {
    pair[][] gdscnt=gds[cnt];
    guide[] resultcnt=result[cnt]=new guide[gdscnt.length];
    for(int i=0; i < gdscnt.length; ++i) {
      pair[] pts=gdscnt[i];
      guide gd=pts[0];
      for(int j=1; j < pts.length-1; ++j)
      	gd=join(gd,pts[j]);
      if(abs(pts[0]-pts[pts.length-1]) < eps)
        gd=gd..cycle;
      else
	gd=join(gd,pts[pts.length-1]);
      resultcnt[i]=gd;
    }
  }
  return result;
}

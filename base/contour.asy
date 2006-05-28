public real eps=10*realEpsilon;
public int nmesh=25;       // default mesh subdivisions

private struct cgd
{
  public pair[] g;	  // nodes
  public bool actv=true;  // is guide active
  public bool exnd=true;  // has the guide been extended
}
  
cgd operator init() {return new cgd;}

private struct segment
{
  public pair a;
  public pair b;
}
  
segment operator init() {return new segment;}

// Case 1: line passes through two vertices of a triangle
private segment case1(pair pt1, pair pt2)
{
  // Will cause a duplicate guide; luckily case1 is very rare
  segment rtrn;
  rtrn.a=pt1;
  rtrn.b=pt2;
  return rtrn;
}

// Case 2: line passes a vertex and a side of a triangle
// (the first vertex passed and the side between the other two)
private segment case2(pair pts0, pair pts1, pair pts2,
		      real vls0, real vls1, real vls2)
{
  pair isect;
  isect=pts1+(pts2-pts1)*abs(vls1/(vls2-vls1));
  segment rtrn;
  rtrn.a=pts0;
  rtrn.b=isect;
  return rtrn;
}

// Case 3: line passes through two sides of a triangle
// (through the sides formed by the first & second, and second & third
// vertices)
private segment case3(pair pts0, pair pts1, pair pts2,
		      real vls0, real vls1, real vls2)
{
  pair isect1,isect2;
  isect1=pts1+(pts2-pts1)*abs(vls1/(vls2-vls1));
  isect2=pts1+(pts0-pts1)*abs(vls1/(vls0-vls1));
  segment rtrn;
  rtrn.a=isect1;
  rtrn.b=isect2;
  return rtrn;
}

// Check if a line passes through a triangle, and draw the required line.
private segment checktriangle(pair pts0, pair pts1, pair pts2,
			      real vls0, real vls1, real vls2)
{  
  //default null return  
  segment dflt; dflt.a=(0,0); dflt.b=(0,0);
  
  if(vls0 < 0) {
    if(vls1 < 0) {
      if(vls2 < 0) return dflt; // nothing to do
      else if(vls2 == 0) return dflt; // nothing to do
      else return case3(pts0,pts2,pts1,vls0,vls2,vls1);
    } else if(vls1 == 0) {
      if(vls2 < 0) return dflt; // nothing to do
      else if(vls2 == 0) return case1(pts1,pts2);
      else return case2(pts1,pts0,pts2,vls1,vls0,vls2);
    } else {
      if(vls2 < 0) return case3(pts0,pts1,pts2,vls0,vls1,vls2);
      else if(vls2 == 0) 
	return case2(pts2,pts0,pts1,vls2,vls0,vls1);
      else return case3(pts2,pts0,pts1,vls2,vls0,vls1);
    } 
  }
  else if(vls0 == 0) {
    if(vls1 < 0) {
      if(vls2 < 0) return dflt; // nothing to do
      else if(vls2 == 0) return case1(pts0,pts2);
      else return case2(pts0,pts1,pts2,vls0,vls1,vls2);
    } else if(vls1 == 0) {
      if(vls2 < 0) return case1(pts0,pts1);
      else if(vls2 == 0) return dflt; // use finer partitioning.
      else return case1(pts0,pts1);
    } else {
      if(vls2 < 0) return case2(pts0,pts1,pts2,vls0,vls1,vls2);
      else if(vls2 == 0) return case1(pts0,pts2);
      else return dflt; // nothing to do
    } 
  } else {
    if(vls1 < 0) {
      if(vls2 < 0) return case3(pts2,pts0,pts1,vls2,vls0,vls1);
      else if(vls2 == 0)
	return case2(pts2,pts0,pts1,vls2,vls0,vls1);
      else return case3(pts0,pts1,pts2,vls0,vls1,vls2);
    }
    else if(vls1 == 0) {
      if(vls2 < 0) return case2(pts1,pts0,pts2,vls1,vls0,vls2);
      else if(vls2 == 0) return case1(pts1,pts2);
      else return dflt; // nothing to do
    } else {
      if(vls2 < 0) return case3(pts0,pts2,pts1,vls0,vls2,vls1);
      else if(vls2 == 0) return dflt; // nothing to do
      else return dflt; // nothing to do
    } 
  }      
}

// check existing guides and adds new segment to them if possible,
// or otherwise store segment as a new guide
private void addseg(segment seg, cgd[] gds)
{ 
  // initialization 
  // search for a path to extend
  for (int i=0; i < gds.length; ++i) {
    cgd gdsi=gds[i];
    if(!gdsi.actv) continue;
    pair[] gd=gdsi.g;
    if(abs(gd[0]-seg.b) < eps) {
      gdsi.g.insert(0,seg.a);
      gdsi.exnd=true; 
      return;
    } else if(abs(gd[gd.length-1]-seg.b) < eps) {
      gdsi.g.push(seg.a);
      gdsi.exnd=true; 
      return;
    } else if(abs(gd[0]-seg.a) < eps) {
      gdsi.g.insert(0,seg.b);
      gdsi.exnd=true;
      return;
    } else if(abs(gd[gd.length-1]-seg.a) < eps) {  
      gdsi.g.push(seg.b);
      gdsi.exnd=true; 
      return;
    }
  }
 
  // in case nothing is found
  cgd segm;
  segm.g=new pair[] {seg.a,seg.b}; 
  gds.push(segm); return;
}

typedef guide interpolate(... guide[]);

// return contour guides computed using a triangle mesh
// f:        function for which we are finding contours
// a,b:      lower left and upper right vertices of rectangle
// c:        contour level
// n,m:      subdivisions on x and y axes (affects accuracy)
// join:     type of interpolation (linear, bezier, etc)
guide[][] contourguides(real[][] f, real[][] midpoint=new real[][],
			pair a, pair b, real[] c, int n=nmesh, int m=n,
			interpolate join=operator --)
{    
  bool midpoints=midpoint.length > 0;
  
  // check if boundaries are good
  if(b.x <= a.x || b.y <= a.y) {
    abort("bad contour domain: make sure the second passed point 
		is above and to the right of the first one.");
  } 

  // array to store guides found so far
  cgd[][] gds=new cgd[c.length][0];

  real ninv=1/n;
  real minv=1/m;
  
  // go over region a rectangle at a time
  for(int col=0; col < n; ++col) {
    real dx=(b.x-a.x)*ninv;
    real x=a.x+col*dx;
    for(int row=0; row < m; ++row) {
      real dy=(b.y-a.y)*minv;
      real y=a.y+row*dy;
      
      // define points
      pair bleft=(x,y);
      pair bright=(x+dx,y);
      pair tleft=(x,y+dy);
      pair tright=(x+dx,y+dy);
      pair middle=0.5*(bleft+tright);
   
      real f00=f[col][row];
      real f01=f[col][row+1];
      real f10=f[col+1][row];
      real f11=f[col+1][row+1];
      
      for(int cnt=0; cnt < c.length; ++cnt) {
	real C=c[cnt];
        real vertdat0=f00-C;  // lower-left vertex
        real vertdat1=f10-C;  // lower-right vertex
        real vertdat2=f01-C;  // upper-left vertex
        real vertdat3=f11-C;  // upper-right vertex

        // optimization: we make sure we don't work with empty rectangles
        int count0=0;
        int count1=0;
        int count2=0;
	
	void check(real vertdat) {
          if(abs(vertdat) < eps) ++count1; 
          else if(vertdat < 0) ++count0;
          else ++count2;
	}
	
	check(vertdat0);
	check(vertdat1);
	check(vertdat2);
	check(vertdat3);

        if(count0 == 4 || count2 == 4) continue;  // nothing to do 
        if((count0 == 3 || count2 == 3) && count1 == 1) continue;

        // evaluate point at middle of rectangle (to set up triangles)
	real vertdat4=midpoints ? midpoint[col][row]-C :
	  0.25*(vertdat0+vertdat1+vertdat2+vertdat3);
      
        segment curseg;
     
        // go through the triangles
        curseg=checktriangle(tleft,tright,middle,
			     vertdat2,vertdat3,vertdat4);
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(tright,bright,middle,
			     vertdat3,vertdat1,vertdat4);
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(bright,bleft,middle,
			     vertdat1,vertdat0,vertdat4);
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(bleft,tleft,middle,
			     vertdat0,vertdat2,vertdat4);
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
      }
    }
    // check which guides are still extendable
    for(int cnt=0; cnt < c.length; ++cnt) {
      cgd[] gdscnt=gds[cnt];
      for(int i=0; i < gdscnt.length; ++i) {
	cgd gdscnti=gdscnt[i];
        if(gdscnti.exnd) gdscnti.exnd=false;
        else gdscnti.actv=false;
      }
    }
  }

  // connect existing paths
  
  // use to reverse an array, omitting the first point
  int[] reverseF(int n) {return sequence(new int(int x){return n-1-x;},n-1);}
  // use to reverse an array, omitting the last point
  int[] reverseL(int n) {return sequence(new int(int x){return n-2-x;},n-1);}
  
  for(int cnt=0; cnt < c.length; ++cnt) {
    cgd[] gdscnt=gds[cnt];
    for(int i=0; i < gdscnt.length; ++i) {
      pair[] gig=gdscnt[i].g;
      int Li=gig.length;
      for(int j=i+1; j < gdscnt.length; ++j) {
        cgd gj=gdscnt[j];
        pair[] gjg=gj.g;
	int Lj=gjg.length;
        if(abs(gig[0]-gjg[0]) < eps) { 
	  gj.g=gjg[reverseF(Lj)];
	  gj.g.append(gig);
          gdscnt.delete(i); 
          --i; 
          break;
        } else if(abs(gig[0]-gjg[Lj-1]) < eps) {
	  gig.delete(0);
	  gjg.append(gig);
          gdscnt.delete(i);
          --i;
          break;
        } else if(abs(gig[Li-1]-gjg[0]) < eps) {
	  gjg.delete(0);
	  gig.append(gjg);
	  gj.g=gig;
          gdscnt.delete(i);
          --i;
          break;
        } else if(abs(gig[Li-1]-gjg[Lj-1]) < eps) {
	  gig.append(gjg[reverseL(Lj)]);
          gj.g=gig;
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
    cgd[] gdscnt=gds[cnt];
    result[cnt]=new guide[gdscnt.length];
    for(int i=0; i < gdscnt.length; ++i) {
      pair[] pts=gdscnt[i].g;
      guide gd=pts[0];
      for(int j=1; j < pts.length-1; ++j)
      	gd=join(gd,pts[j]);
      if(abs(pts[0]-pts[pts.length-1]) < eps)
        gd=gd..cycle;
      else
	gd=join(gd,pts[pts.length-1]);
      result[cnt][i]=gd;
    }
  }
  return result;
}

guide[][] contourguides(real f(real, real), pair a, pair b,
			real[] c, int n=nmesh, int m=n,
			interpolate join=operator --)
{    
  // evaluate function at points and midpoints
  real[][] dat=new real[n+1][m+1];
  real[][] midpoint=new real[n+1][m+1];
  for(int i=0; i <= n; ++i) {
    for(int j=0; j <= m; ++j) {
      dat[i][j]=f(interp(a.x,b.x,i/n),interp(a.y,b.y,j/m));
      midpoint[i][j]=f(interp(a.x,b.x,(i+0.5)/n),interp(a.y,b.y,(j+0.5)/m));
    }
  }
  return contourguides(dat,midpoint,a,b,c,n,m,join);
}
  
void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real[] c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p(real))
{
  guide[][] g;
  g=contourguides(f,a,b,c,n,m,join);
  for(int cnt=0; cnt < c.length; ++cnt)
    for(int i=0; i < g[cnt].length; ++i)
      draw(pic,g[cnt][i],p(c[cnt]));
  /*
  for(int cnt=0; cnt < c.length; ++cnt)
    for(int i=0; i < g[cnt].length; ++i)
      label(pic,Label((string) c[cnt],align=(0,0),UnFill),g[cnt][i],p(c[cnt]));
  */
}

void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p(real))
{
  contour(pic,f,a,b,new real[] {c},n,m,join,p);
}

void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real[] c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p=currentpen)
{
  contour(pic,f,a,b,c,n,m,join,new pen(real) {return p;});
}

void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p=currentpen)
{
  contour(pic,f,a,b,new real[] {c},n,m,join,new pen(real) {return p;});
}

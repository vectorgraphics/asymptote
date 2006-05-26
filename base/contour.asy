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
private segment case2(pair[] pts, real[] vls)
{
  pair isect;
  isect=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  segment rtrn;
  rtrn.a=pts[0];
  rtrn.b=isect;
  return rtrn;
}

// Case 3: line passes through two sides of a triangle
// (through the sides formed by the first & second, and second & third
// vertices)
private segment case3(pair[] pts, real[] vls)
{
  pair isect1,isect2;
  isect1=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  isect2=pts[1]+(pts[0]-pts[1])*fabs(vls[1]/(vls[0]-vls[1]));
  segment rtrn;
  rtrn.a=isect1;
  rtrn.b=isect2;
  return rtrn;
}

// Check if a line passes through a triangle, and draw the required line.
private segment checktriangle(pair[] pts, real[] vls)
{  
  //default null return  
  segment dflt; dflt.a=(0,0); dflt.b=(0,0);
  
  if(vls[0] < 0) {
    if(vls[1] < 0) {
      if(vls[2] < 0) return dflt;          // nothing to do
      else if(vls[2] == 0) return dflt;  // nothing to do
      else return case3(new pair[] {pts[0],pts[2],pts[1]},
			new real[] {vls[0],vls[2],vls[1]}); // case 3
    }
    else if(vls[1] == 0) {
      if(vls[2] < 0) return dflt;       // nothing to do
      else if(vls[2] == 0) return case1(pts[1],pts[2]); // case 1
      else return case2(new pair[] {pts[1],pts[0],pts[2]},
			new real[] {vls[1],vls[0],vls[2]}); // case 2
    }
    else {
      if(vls[2] < 0) return case3(new pair[] {pts[0],pts[1],pts[2]},
				  new real[] {vls[0],vls[1],vls[2]}); // case 3
      else if(vls[2] == 0) 
	return case2(new pair[] {pts[2],pts[0],pts[1]},
		     new real[] {vls[2],vls[0],vls[1]}); // case 2
      else return case3(new pair[] {pts[2],pts[0],pts[1]},
			new real[] {vls[2],vls[0],vls[1]}); // case 3
    } 
  }
  else if(vls[0] == 0) {
    if(vls[1] < 0) {
      if(vls[2] < 0) return dflt; // nothing to do
      else if(vls[2] == 0) return case1(pts[0],pts[2]); // case 1
      else return case2(new pair[] {pts[0],pts[1],pts[2]},
			new real[] {vls[0],vls[1],vls[2]}); // case 2
    }
    else if(vls[1] == 0) {
      if(vls[2] < 0) return case1(pts[0],pts[1]); // case 1
      else if(vls[2] == 0) return dflt; // use finer partitioning.
      else return case1(pts[0],pts[1]); // case 1
    }
    else {
      if(vls[2] < 0) return case2(new pair[] {pts[0],pts[1],pts[2]},
				  new real[] {vls[0],vls[1],vls[2]}); // case 2
      else if(vls[2] == 0) return case1(pts[0],pts[2]); // case 1
      else return dflt; // nothing to do
    } 
  }
  else {
    if(vls[1] < 0) {
      if(vls[2] < 0) return case3(new pair[] {pts[2],pts[0],pts[1]},
				  new real[] {vls[2],vls[0],vls[1]}); // case 3
      else if(vls[2] == 0)
	return case2(new pair[] {pts[2],pts[0],pts[1]},
		     new real[] {vls[2],vls[0],vls[1]}); // case 2
      else return case3(new pair[] {pts[0],pts[1],pts[2]},
			new real[] {vls[0],vls[1],vls[2]}); // case 3
    }
    else if(vls[1] == 0) {
      if(vls[2] < 0) return case2(new pair[] {pts[1],pts[0],pts[2]},
				  new real[] {vls[1],vls[0],vls[2]}); // case 2
      else if(vls[2] == 0) return case1(pts[1],pts[2]); // case 1
      else return dflt; // nothing to do
    }
    else {
      if(vls[2] < 0) return case3(new pair[] {pts[0],pts[2],pts[1]},
				  new real[] {vls[0],vls[2],vls[1]}); // case 3
      else if(vls[2] == 0) return dflt; // nothing to do
      else return dflt; // nothing to do
    } 
  }      
}

// check existing guides and adds new segment to them if possible,
// or otherwise store segment as a new guide
private void addseg(segment seg, cgd[] gds)
{ 
  // initialization 
  if (gds.length == 0) { 
    cgd segm; segm.g.push(seg.a); segm.g.push(seg.b); 
    gds.push(segm); return;
  }

  // searching for a path to extend
  int i;  
  for (i=0; i < gds.length; ++i) {
    if(!gds[i].actv) continue;
    pair[] gd=gds[i].g;
    if(length(gd[0]-seg.b) < eps) {
      pair[] toadd=new pair[]{seg.a};
      toadd.append(gd);
      gds[i].g=toadd;
      gds[i].exnd=true; 
      return;
    }
    else if(length(gd[gd.length-1]-seg.b) < eps) {
      gds[i].g.push(seg.a);
      gds[i].exnd=true; 
      return;
    }
    else if(length(gd[0]-seg.a) < eps) {
      pair[] toadd=new pair[]{seg.b};
      toadd.append(gd);
      gds[i].g=toadd;
      gds[i].exnd=true;
      return;
    }
    else if(length(gd[gd.length-1]-seg.a) < eps) {  
      gds[i].g.push(seg.b);
      gds[i].exnd=true; 
      return;
    }
  }
 
  // in case nothing is found
  cgd segm; segm.g.push(seg.a); segm.g.push(seg.b); 
  gds.push(segm); 
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

  // go over region a rectangle at a time
  for(int col=0; col < n; ++col) {
    for(int row=0; row < m; ++row) {
      for(int cnt=0; cnt < c.length; ++cnt) {
        real[] vertdat=new real[5];   // neg-below, 0 -at, pos-above;
        vertdat[0]=f[col][row]-c[cnt];      // lower-left vertex
        vertdat[1]=f[col+1][row]-c[cnt];    // lower-right vertex
        vertdat[2]=f[col][row+1]-c[cnt];    // upper-left vertex
        vertdat[3]=f[col+1][row+1]-c[cnt];  // upper-right vertex

        // optimization: we make sure we don't work with empty rectangles
        int[] count=new int[3]; count[0]=0; count[1]=0; count[2]=0; 
        for(int i=0; i < 4; ++i) {
          if(fabs(vertdat[i]) < eps)++count[1]; 
          else if(vertdat[i] < 0)++count[0];
          else++count[2];
	}
        if((count[0] == 4) || (count[2] == 4)) continue;  // nothing to do 
        if((count[0] == 3 && count[1] == 1) || 
	   (count[2] == 3 && count[1] == 1)) continue;

        // evaluates point at middle of rectangle(to set up triangles)

	vertdat[4]=midpoints ? midpoint[col][row]-c[cnt] :
	  0.25*(vertdat[0]+vertdat[1]+vertdat[2]+vertdat[3]);
      
        // define points
        pair bleft=(a.x+(b.x-a.x)*col/n,a.y+(b.y-a.y)*row/m);
        pair bright=(a.x+(b.x-a.x)*(col+1)/n,a.y+(b.y-a.y)*row/m);
        pair tleft=(a.x+(b.x-a.x)*col/n,a.y+(b.y-a.y)*(row+1)/m);
        pair tright=(a.x+(b.x-a.x)*(col+1)/n,a.y+(b.y-a.y)*(row+1)/m);
        pair middle=(a.x+(b.x-a.x)*(col+1/2)/n,
		     a.y+(b.y-a.y)*(row+1/2)/m);
   
        segment curseg;
     
        // go through the triangles
        curseg=checktriangle(new pair[] {tleft,tright,middle},
			     new real[] {vertdat[2],vertdat[3],vertdat[4]});
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(new pair[] {tright,bright,middle},
			     new real[] {vertdat[3],vertdat[1],vertdat[4]});
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(new pair[] {bright,bleft,middle},
			     new real[] {vertdat[1],vertdat[0],vertdat[4]});
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
        curseg=checktriangle(new pair[] {bleft,tleft,middle},
			     new real[] {vertdat[0],vertdat[2],vertdat[4]});
        if(length(curseg.a-curseg.b) > eps) addseg(curseg,gds[cnt]);
      }
    }
    // checks which guides are still extendable
    for(int cnt=0; cnt < c.length; ++cnt) {
      for(int i=0; i < gds[cnt].length; ++i) {
        if(gds[cnt][i].exnd) gds[cnt][i].exnd=false;
        else gds[cnt][i].actv=false;
      }
    }
  }

  // connect existing paths
  for(int cnt=0; cnt < c.length; ++cnt) {
    for(int i=0; i < gds[cnt].length; ++i) {
      for(int j=i+1; j < gds[cnt].length; ++j) {
        pair[] gi=gds[cnt][i].g;
        pair[] gj=gds[cnt][j].g;
        if     (length(gi[0]-gj[0]) < eps) { 
	  pair[] np;
          for(int q=gj.length-1; q > 0; --q)
	    np.push(gj[q]);
	  np.append(gi);
          gds[cnt][j].g=np;
          gds[cnt].delete(i); 
          --i; 
          break;
        }
        else if(length(gi[0]-gj[gj.length-1]) < eps) { 
	  for(int q=1; q < gi.length; ++q)
	    gj.push(gi[q]);
          gds[cnt][j].g=gj;
          gds[cnt].delete(i);
          --i;
          break;
        }
        else if(length(gi[gi.length-1]-gj[0]) < eps) { 
	  for(int q=1; q < gj.length; ++q)
	    gi.push(gj[q]);
          gds[cnt][j].g=gi;
          gds[cnt].delete(i);
          --i;
          break;
        }
        else if(length(gi[gi.length-1]-gj[gj.length-1]) < eps) { 
	  pair[] np;
          for(int q=gj.length-2; q > -1; --q)
	    np.push(gj[q]);      
	  gi.append(np);  
          gds[cnt][j].g=gi;
          gds[cnt].delete(i);
          --i;
          break;
        } 
      }
    }
  }

  // setting up return value
  guide[][] result=new guide[c.length][0];
  for(int cnt=0; cnt < c.length; ++cnt) {
    result[cnt]=new guide[gds[cnt].length];
    for(int i=0; i < gds[cnt].length; ++i) {
      pair[] pts=gds[cnt][i].g;
      guide gd=pts[0];
      for(int j=1; j < pts.length; ++j)
      	gd=join(gd,pts[j]);
      if(length(pts[0]-pts[pts.length-1]) < eps)
        gd=gd--cycle;
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
  if(false)  
  for(int cnt=0; cnt < c.length; ++cnt)
    for(int i=0; i < g[cnt].length; ++i)
      label(pic,Label((string) c[cnt],align=(0,0),UnFill),g[cnt][i],p(c[cnt]));
}

void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real[] c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p=currentpen)
{
  contour(pic,f,a,b,c,n,m,join,p);
}

void contour(picture pic=currentpicture, real f(real, real),
	     pair a, pair b, real c, int n=nmesh,
	     int m=n, interpolate join=operator --, pen p=currentpen)
{
  contour(pic,f,a,b,new real[] {c},n,m,join,
	  new pen(real) {return currentpen;});
}

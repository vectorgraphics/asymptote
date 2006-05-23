public real eps=10*realEpsilon;
public int xndefault=25;       // cuts on x axis
public int yndefault=25;       // cuts on y axis

private struct cgd
{
  public guide g;
  public bool actv=true;  // is guide active
  public bool exnd=true;  // has the guide been extended
}
cgd operator init(){return new cgd;}

// Case 1: line passes through two vertices of a triangle
private guide case1(pair pt1, pair pt2)
{
  // WILL cause a bug due to repetition; luckily case1 is very rare
  return pt1--pt2;
}

// Case 2: line passes a vertex and a side of a triangle
// (the first vertex passed and the side between the other two)
private guide case2(pair[] pts, real[] vls)
{
  pair isect;
  isect=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  return pts[0]--isect;
}

// Case 3: line passes through two sides of a triangle
// (through the sides formed by the first & second, and second & third
// vertices)
private guide case3(pair[] pts, real[] vls)
{
  pair isect1,isect2;
  isect1=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  isect2=pts[1]+(pts[0]-pts[1])*fabs(vls[1]/(vls[0]-vls[1]));
  return isect1--isect2;
}

// Check if a line passes through a triangle, and draw the required line.
private guide checktriangle(pair[] pts, real[] vls)
{  
  if(vls[0] < 0) {
    if(vls[1] < 0) {
      if(vls[2] < 0) return nullpath;          // nothing to do
      else if(vls[2] == 0) return nullpath;  // nothing to do
      else return case3(new pair[] {pts[0],pts[2],pts[1]},
		 new real[] {vls[0],vls[2],vls[1]}); // case 3
    }
    else if(vls[1] == 0) {
      if(vls[2] < 0) return nullpath;       // nothing to do
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
      if(vls[2] < 0) return nullpath; // nothing to do
      else if(vls[2] == 0) return case1(pts[0],pts[2]); // case 1
      else return case2(new pair[] {pts[0],pts[1],pts[2]},
		 new real[] {vls[0],vls[1],vls[2]}); // case 2
    }
    else if(vls[1] == 0) {
      if(vls[2] < 0) return case1(pts[0],pts[1]); // case 1
      else if(vls[2] == 0) return nullpath; // use finer partitioning.
      else return case1(pts[0],pts[1]); // case 1
    }
    else {
      if(vls[2] < 0) return case2(new pair[] {pts[0],pts[1],pts[2]},
			   new real[] {vls[0],vls[1],vls[2]}); // case 2
      else if(vls[2] == 0) return case1(pts[0],pts[2]); // case 1
      else return nullpath; // nothing to do
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
      else return nullpath; // nothing to do
    }
    else {
      if(vls[2] < 0) return case3(new pair[] {pts[0],pts[2],pts[1]},
			   new real[] {vls[0],vls[2],vls[1]}); // case 3
      else if(vls[2] == 0) return nullpath; // nothing to do
      else return nullpath; // nothing to do
    } 
  }      
}

// removes an element from an array
private void pop(cgd[] gds, int idx)
{
  for(int i=idx+1; i < gds.length; ++i) gds[i-1]=gds[i];
  gds.pop();
  return;
}


// checks existing guides and adds new segment to them if possible,
// or otherwise stores segment as a new guide
private void addseg(guide seg, cgd[] gds)
{ 
  // initialization 
  if (gds.length == 0){ 
    cgd segm; segm.g=seg; gds.push(segm); return;
  }

  // searching for a path to extend
  int  i;  
  for (i=0; i < gds.length; ++i) {
    if(!gds[i].actv) continue;
    if(length(point(gds[i].g,0)-point(seg,size(seg))) < eps) {
      gds[i].g=seg--gds[i].g;
      gds[i].exnd=true; 
      return;
    }
    else if(length(point(gds[i].g,size(gds[i].g))-
			point(seg,size(seg))) < eps) {
      gds[i].g=gds[i].g--reverse(seg); 
      gds[i].exnd=true; 
      return;
    }
    else if(length(point(gds[i].g,0)-point(seg,0)) < eps) {
      gds[i].g=reverse(seg)--gds[i].g;
      gds[i].exnd=true;
      return;
    }
    else if(length(point(gds[i].g,size(gds[i].g))-point(seg,0)) < eps) {  
      gds[i].g=gds[i].g--seg;
      gds[i].exnd=true; 
      return;
    }
  }
 
  // in case nothing is found
  if(i == gds.length){
    cgd segm; segm.g=seg; gds.push(segm); return;
  }

  return;
}

/* contouring using a triangle mesh returns guides
 *func:        function for which we are finding contours
 *cl:          contour level
 *ll,ur:       lower left and upper right vertices of rectangle
 *xn,yn:       cuts on each axis (i.e. accuracy)
 */
guide[][] contourguides(real func(real, real), real[] cl,
			pair ll, pair ur, int xn=xndefault,
			int yn=yndefault)
{    
  // check if boundaries are good
  if(ur.x <= ll.x || ur.y <= ll.y) {
    abort("bad contour domain: make sure the second passed point 
		is above and to the right of the first one.");
  } 

  // evaluate function at points
  real[][] dat=new real[xn+1][yn+1];
  for(int i=0; i < xn+1; ++i) {
    for(int j=0; j < yn+1; ++j) {
      dat[i][j]=func(ll.x+(ur.x-ll.x)*i/xn,ll.y+(ur.y-ll.y)*j/yn);
    }
  } 

  // array to store guides found so far
  cgd[][] gds=new cgd[cl.length][0];

  // go over region a rectangle at a time
  for(int col=0; col < xn; ++col) {
    for(int row=0; row < yn; ++row) {
      for(int cnt=0; cnt < cl.length; ++cnt) {
        real[] vertdat=new real[5];   // neg-below, 0 -at, pos-above;
        vertdat[0]=(dat[col][row]-cl[cnt]);      // lower-left vertex
        vertdat[1]=(dat[col+1][row]-cl[cnt]);    // lower-right vertex
        vertdat[2]=(dat[col][row+1]-cl[cnt]);    // upper-left vertex
        vertdat[3]=(dat[col+1][row+1]-cl[cnt]);  // upper-right vertex

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
        real midpt=func(ll.x+(ur.x-ll.x)*(col+1/2)/xn,ll.y+
			(ur.y-ll.y)*(row+1/2)/yn);
	vertdat[4]=(midpt-cl[cnt]);                      // midpoint  
      
        // define points
        pair bleft=(ll.x+(ur.x-ll.x)*col/xn,ll.y+(ur.y-ll.y)*row/yn);
        pair bright=(ll.x+(ur.x-ll.x)*(col+1)/xn,ll.y+(ur.y-ll.y)*row/yn);
        pair tleft=(ll.x+(ur.x-ll.x)*col/xn,ll.y+(ur.y-ll.y)*(row+1)/yn);
        pair tright=(ll.x+(ur.x-ll.x)*(col+1)/xn,ll.y+(ur.y-ll.y)*(row+1)/yn);
        pair middle=(ll.x+(ur.x-ll.x)*(col+1/2)/xn,
		     ll.y+(ur.y-ll.y)*(row+1/2)/yn);
   
        guide curseg;
     
        // go through the triangles
        curseg=checktriangle(new pair[] {tleft,tright,middle},
		      new real[] {vertdat[2],vertdat[3],vertdat[4]});
        if(arclength(curseg) > eps) addseg(curseg, gds[cnt]);
        curseg=checktriangle(new pair[] {tright,bright,middle},
		      new real[] {vertdat[3],vertdat[1],vertdat[4]});
        if(arclength(curseg) > eps) addseg(curseg, gds[cnt]);
        curseg=checktriangle(new pair[] {bright,bleft,middle},
		      new real[] {vertdat[1],vertdat[0],vertdat[4]});
        if(arclength(curseg) > eps) addseg(curseg, gds[cnt]);
        curseg=checktriangle(new pair[] {bleft,tleft,middle},
		      new real[] {vertdat[0],vertdat[2],vertdat[4]});
        if(arclength(curseg) > eps) addseg(curseg, gds[cnt]);
      }
    }
    for(int cnt=0; cnt < cl.length; ++cnt) {
      for(int i=0; i < gds[cnt].length; ++i) {
        if(gds[cnt][i].exnd) gds[cnt][i].exnd=false;
        else gds[cnt][i].actv=false;
      }
    }
  }

  // connect existing paths
  for(int cnt=0; cnt < cl.length; ++cnt) {
    for(int i=0; i < gds[cnt].length; ++i) {
      for(int j=i+1; j < gds[cnt].length; ++j) {
        if     (length(point(gds[cnt][i].g,0)-
			point(gds[cnt][j].g,0)) < eps) { 
          gds[cnt][j].g=reverse(gds[cnt][j].g)--gds[cnt][i].g;
          pop(gds[cnt],i); 
          --i; 
          break;
        }
        else if(length(point(gds[cnt][i].g,0)-
			point(gds[cnt][j].g,size(gds[cnt][j].g))) < eps) { 
          gds[cnt][j].g=gds[cnt][j].g--gds[cnt][i].g;
          pop(gds[cnt],i);
          --i;
          break;
        }
        else if(length(point(gds[cnt][i].g,size(gds[cnt][i].g))-
			point(gds[cnt][j].g,0)) < eps) {    
          gds[cnt][j].g=gds[cnt][i].g--gds[cnt][j].g;
          pop(gds[cnt],i);
          --i;
          break;
        }
        else if(length(point(gds[cnt][i].g,size(gds[cnt][i].g))-
			point(gds[cnt][j].g,size(gds[cnt][j].g))) < eps) { 
          gds[cnt][j].g=gds[cnt][i].g--reverse(gds[cnt][j].g);
          pop(gds[cnt],i);
          --i;
          break;
        } 
      }
    }
  }

  // closes cyclic guides
  for(int cnt=0; cnt < cl.length; ++cnt) {
    for(int i=0; i < gds[cnt].length; ++i) {
      if(length(point(gds[cnt][i].g,0)-
		point(gds[cnt][i].g,size(gds[cnt][i].g))) < eps)
        gds[cnt][i].g=gds[cnt][i].g--cycle;
    }
  }
  
  // setting up return value
  guide[][] result=new guide[cl.length][0];
  for(int cnt=0; cnt < cl.length; ++cnt) {
    result[cnt]=new guide[gds[cnt].length];
    for(int i=0; i < gds[cnt].length; ++i) {
      result[cnt][i]=gds[cnt][i].g;
    }
  }

  return result;
}


void contour(picture pic=currentpicture, Label L="", real func(real, real),
	     real[] cl, pair ll, pair ur, int xn=xndefault,
	     int yn=yndefault, pen[] p)
{
  guide[][] g;
  g=contourguides(func,cl,ll,ur,xn,yn);
  for(int cnt=0; cnt < cl.length; ++cnt) {
    for(int i=0; i < g[cnt].length; ++i)
      draw(pic,L,g[cnt][i],p[cnt]);
  }
  return;
}

 
void contour(picture pic=currentpicture, Label L="", real func(real, real),
	     real[] cl, pair ll, pair ur, int xn=xndefault,
	     int yn=yndefault, pen p=currentpen)
{
  pen[] pp=new pen[cl.length];
  for(int i=0; i < cl.length; ++i) pp[i]=p;
  contour(pic,L,func,cl,ll,ur,xn,yn,pp);
}


void contour(picture pic=currentpicture, Label L="", real func(real, real),
	     real cl, pair ll, pair ur, int xn=xndefault,
	     int yn=yndefault, pen p=currentpen)
{
  contour(pic,L,func,new real[] {cl},ll,ur,xn,yn,new pen[]{p});
}

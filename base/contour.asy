import math;

public real eps=10*realEpsilon;
public int  xndefault=100;       //cuts on x axis
public int  yndefault=100;       //cuts on y axis
public bool griddflt=false;      //whether or not to draw grid

//draws case1
//case1: line passes through two vertices of a triangle
private void case1 (picture pic, pen penn, pair pt1, pair pt2)
{
  draw(pic,pt1--pt2,penn);
  return;
}

//draws case2
//case2: line passes a vertex and a side of a triangle
//(the first vertex passed and the side between the other two)
private void case2 (picture pic, pen penn, pair[] pts, real[] vls)
{
  pair isect;
  isect=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  draw(pic,pts[0]--isect,penn);
  return;
}

//draws case3
//case3: line passes through two sides of a triangle
//(through the sides formed by the first and second, and second and third vertices)
private void case3 (picture pic, pen penn, pair[] pts, real[] vls)
{
  pair isect1, isect2;
  isect1=pts[1]+(pts[2]-pts[1])*fabs(vls[1]/(vls[2]-vls[1]));
  isect2=pts[1]+(pts[0]-pts[1])*fabs(vls[1]/(vls[0]-vls[1]));
  draw(pic,isect1--isect2,penn);
  return;
}


//private routine that takes a triangle, checks if the line passes through it, and 
//draws the required line through the triangle
private void checktriangle (picture pic, pen penn, pair[] pts, real[] vls)
{
  if(vls[0] < 0){
    if(vls[1] < 0){
      if(vls[2] < 0) return;          //nothing to do
      else if(vls[2] == 0) return;  //nothing to do
      else case3(pic,penn,new pair[]{pts[0],pts[2],pts[1]}, new real[]{vls[0],vls[2],vls[1]}); //case3
    }
    else if(vls[1] == 0){
      if(vls[2] < 0) return;       //nothing to do
      else if(vls[2] == 0) case1(pic,penn,pts[1], pts[2]); //case1
      else case2(pic,penn,new pair[]{pts[1], pts[0], pts[2]}, new real[]{vls[1],vls[0],vls[2]}); //case2
    }
    else {
      if(vls[2] < 0) case3(pic,penn,new pair[]{pts[0],pts[1],pts[2]}, new real[]{vls[0],vls[1],vls[2]}); //case3
      else if(vls[2] == 0) case2(pic,penn,new pair[]{pts[2],pts[0],pts[1]}, new real[]{vls[2],vls[0],vls[1]}); //case2
      else case3(pic,penn,new pair[]{pts[2],pts[0],pts[1]}, new real[]{vls[2],vls[0],vls[1]}); //case3
    } 
  }
  else if(vls[0] == 0){
    if(vls[1] < 0){
      if(vls[2] < 0) return; //nothing to do
      else if(vls[2] == 0) case1(pic,penn,pts[0], pts[2]); //case1
      else case2(pic,penn,new pair[]{pts[0], pts[1], pts[2]}, new real[]{vls[0],vls[1],vls[2]}); //case2
    }
    else if(vls[1] == 0){
      if(vls[2] < 0) case1(pic,penn,pts[0],pts[1]); //case1
      else if(vls[2] == 0) return; //all three vertices intersected - very rare. if problematic, use finer partitioning.      
      else case1(pic,penn,pts[0],pts[1]); //case1
    }
    else {
      if(vls[2] < 0) case2(pic,penn,new pair[]{pts[0], pts[1], pts[2]}, new real[]{vls[0],vls[1],vls[2]}); //case2
      else if(vls[2] == 0) case1(pic,penn,pts[0], pts[2]); //case1
      else return; //nothing to do
    } 
  }
  else {
    if(vls[1] < 0){
      if(vls[2] < 0) case3(pic,penn,new pair[]{pts[2],pts[0],pts[1]}, new real[]{vls[2],vls[0],vls[1]}); //case3
      else if(vls[2] == 0) case2(pic,penn,new pair[]{pts[2], pts[0], pts[1]}, new real[]{vls[2],vls[0],vls[1]}); //case2
      else case3(pic,penn,new pair[]{pts[0],pts[1],pts[2]}, new real[]{vls[0],vls[1],vls[2]}); //case3
    }
    else if(vls[1] == 0){
      if(vls[2] < 0) case2(pic,penn,new pair[]{pts[1], pts[0], pts[2]}, new real[]{vls[1],vls[0],vls[2]}); //case2
      else if(vls[2] == 0) case1(pic,penn,pts[1],pts[2]); //case1
      else return; //nothing to do
    }
    else {
      if(vls[2] < 0) case3(pic,penn,new pair[]{pts[0],pts[2],pts[1]}, new real[]{vls[0],vls[2],vls[1]}); //case3
      else if(vls[2] == 0) return; //nothing to do
      else return; //nothing to do
    } 
  }      
}


/* contouring using a triangle mesh
*  pic         - picture
*  func        - function for which we are finding contours
*  cl          - contour level
*  x0,x1,y0,y1 - vertices of rectangle on which we work
*  xn,yn       - cuts on each axis (i.e. accuracy)
*  drawgrid    - whether or not to draw the grid (!!TO DECIDE - rectangles, or triangles, or not include at all?)
*  p            - pen
*/
void contour(picture pic=currentpicture, real func(real, real), real cl,
	     real x0, real x1, real y0, real y1, int xn=xndefault,
	     int yn=yndefault, bool drawgrid=griddflt, pen p=currentpen)
{    
  //check if boundaries are good
  if( (x0 == x1) || (y0 == y1) ) abort("bad area of contouring - give distinct points on each axis");
  if(x1 < x0) {real tmp=x0; x0=x1; x1=tmp;}  //x1 should be the larger coordinate
  if(y1 < y0) {real tmp=y0; y0=y1; y1=tmp;}  

  //evaluate function at points
  real[][] dat=new real[xn+1][yn+1];
  for (int i=0; i < xn+1; ++i){
    for (int j=0; j < yn+1; ++j){
      dat[i][j]=func(x0+(x1-x0)*i/xn, y0+(y1-y0)*j/yn);
    }
  } 
  
  //go over region a rectangle at a time
  for (int col=0; col < xn; ++col){
    for (int row=0; row < yn; ++row){ 
      real[] vertdat=new real[5];   //neg - below, 0 -at, pos - above;
      vertdat[0]=(dat[col][row]-cl);      //lower-left vertex
      vertdat[1]=(dat[col+1][row]-cl);    //lower-right vertex
      vertdat[2]=(dat[col][row+1]-cl);    //upper-left vertex
      vertdat[3]=(dat[col+1][row+1]-cl);  //upper-right vertex

      //optimization: the following check makes sure we don't work with empty rectangles
      int[] count=new int[3]; count[0]=0; count[1]=0; count[2]=0; 
      for (int i=0; i < 4; ++i){
        if( fabs(vertdat[i]) < eps) ++count[1]; 
        else if(vertdat[i] < 0) ++count[0];
        else ++count[2];
      }
      if((count[0] == 4) || (count[2] == 4)) continue;  //nothing to do - line doesn't pass through this rectangle
      if((count[0] == 3 && count[1] == 1) || (count[2] == 3 && count[1] == 1)) continue;

      //evaluates point at middle of rectangle (to set up triangles)
      real midpt=func(x0+(x1-x0)*(col+1/2)/xn,y0+(y1-y0)*(row+1/2)/yn); 
      vertdat[4]=(midpt-cl);                      //midpoint  
      
      //define points
      pair bleft= (x0+(x1-x0)*col/xn,      y0+(y1-y0)*row/yn);
      pair bright=(x0+(x1-x0)*(col+1)/xn,  y0+(y1-y0)*row/yn);
      pair tleft= (x0+(x1-x0)*col/xn,      y0+(y1-y0)*(row+1)/yn);
      pair tright=(x0+(x1-x0)*(col+1)/xn,  y0+(y1-y0)*(row+1)/yn);
      pair middle=(x0+(x1-x0)*(col+1/2)/xn,y0+(y1-y0)*(row+1/2)/yn);

      //go through the triangles
      checktriangle(pic,p,new pair[]{tleft, tright, middle},new real[]{vertdat[2],vertdat[3],vertdat[4]});
      checktriangle(pic,p,new pair[]{tright, bright, middle},new real[]{vertdat[3],vertdat[1],vertdat[4]});
      checktriangle(pic,p,new pair[]{bright, bleft, middle},new real[]{vertdat[1],vertdat[0],vertdat[4]});
      checktriangle(pic,p,new pair[]{bleft, tleft, middle},new real[]{vertdat[0],vertdat[2],vertdat[4]});

      //if flag is on, draw grid
      if(drawgrid){
        draw(bleft--bright--tright--tleft--cycle);
        draw(bleft--middle);draw(bright--middle);draw(tright--middle);draw(tleft--middle);
      }
    }
  }
  return;
}

import solids;
import tube;
import graph3;
import palette;
size(8cm);

currentprojection=perspective(
camera=(13.3596389245356,8.01038090435314,14.4864483364785),
up=(-0.0207054323419367,-0.00472438375047319,0.0236460907598947),
target=(-1.06042550499095,2.68154529985845,0.795007562120261));

defaultpen(fontsize(6pt));

// draw coordinates and frames 
// axis1 is defined by z axis of TBase
// axis2 is defined by z axis of TEnd
void DrawFrame(transform3 TBase, transform3 TEnd, string s)
{
  triple p1,v1,p2,v2;
  p1=TBase*O;
  v1=TBase*Z-p1;
  p2=TEnd*O;
  v2=TEnd*Z-p2;
  triple n=cross(v1,v2);

  real[][] A=
    {
      {v1.x,-v2.x,-n.x},
      {v1.y,-v2.y,-n.y},
      {v1.z,-v2.z,-n.z}
    };

  triple vb=p2-p1;

  real[] b={vb.x,vb.y,vb.z};
    
  // Get the extention along vector v1 and v2, 
  // so, we can get the common normal between two axis
  real[] x=solve(A,b);

  real s1=x[0];
  real s2=x[1];
    
  // get foot of a perpendicular on both axies
  triple foot1=p1+s1*v1;
  triple foot2=p2+s2*v2;
	
  // draw two axis
  triple axis_a,axis_b;
  axis_a=p1+s1*v1*1.5;
  axis_b=p1-s1*v1*1.5;
  draw(axis_a--axis_b);
	
  axis_a=p2+s2*v2*1.5;
  axis_b=p2-s2*v2*1.5;
  draw(axis_a--axis_b);
 
  // draw "a"(common normal) 
  draw(Label("$a_{"+s+"}$"),foot1--foot2,linewidth(1pt)); 

  // draw the coordinates frame
  triple dx,dy,dz,org;
  real length=0.8;
    
  org=foot1;
  dx =length*unit(foot2-foot1); // define the x axis of the frame on "a"
  dz =length*unit(v1);          // define the z axis which is along axis1
  dy =length*unit(cross(dz,dx));
	
  draw(Label("$X_{"+s+"}$",1,align=-dy-dz),org--(org+dx),red+linewidth(1.5pt),
       Arrow3(8));
  draw(Label("$Y_{"+s+"}$",1,align=2dy-dz-dx),org--(org+dy), 
       green+linewidth(1.5pt),	Arrow3(8));
  draw(Label("$Z_{"+s+"}$",1,align=-2dx-dy),org--(org+dz),
       blue+linewidth(1.5pt),	Arrow3(8));
    
  dot(Label("$O_{"+s+"}$",align =-dx-dz,black),org,black); // origin
           
}

void DrawLink(transform3 TBase, transform3 TEnd, pen objStyle,string s)
{
  real h=1;
  real r=0.5;
  path3 generator=(0.5*r,0,h)--(r,0,h)--(r,0,0)--(0.5*r,0,0);
  revolution vase=revolution(O,generator,0,360);
  surface objSurface=surface(vase);
    
  render render=render(merge=true);

  // draw two cylinders
  draw(TBase*objSurface,objStyle,render);
  draw(TEnd*shift((0,0,-h+1e-5))*objSurface,objStyle,render);
	
  // draw the link between two cylinders
  triple pStart=TBase*(0.5*h*Z);
  triple pEnd  =TEnd*(-0.5*h*Z);
  triple pControl1=0.25*(pEnd-pStart)+TBase*(0,0,h);
  triple pControl2=-0.25*(pEnd-pStart)+TEnd*(0,0,-h);
  path3 p=pStart..controls pControl1 and pControl2..pEnd;
  draw(tube(p,scale(0.2)*unitsquare),objStyle,render);   
}

// t1 and t2 define the starting frame and ending frame of the first link(i-1)
transform3 t1=shift((0,0,1));
transform3 t2=shift((0,0,-1))*rotate(-20,Y)*shift((0,3,2));
// as, the two links were connected, so t2 is also the starting frame of link(i)
// t3 defines the ending frame of link(i) 
transform3 t3=t2*rotate(40,Z)*shift((0,3,1.5))*rotate(-15,Y)*shift(-1.5*Z);

// draw link(i-1)
DrawLink(t1,t2,palegreen,"i-1");
DrawFrame(t1,t2,"i-1");
// draw link(i)
DrawLink(t2,t3,lightmagenta,"i");
DrawFrame(t2,t3,"i");


// draw angle alpha, which is the angle between axis(i-1) and axis(i)
triple p0=(0,0,-1);
triple p1=(0,0,2.3);
triple p2=shift((0,0,-1))*rotate(-20,Y)*(0,0,4);
draw(p0--p2,cyan);
draw("$\alpha_{i-1}$",arc(p0,p1,p2,Y,CW),ArcArrow3(3));


// draw angle theta, which is the angle between a_i and a_{i-1}
transform3 tx=shift((0,0,-1))*rotate(-20,Y)*shift((0,3,0));
p0=tx*O;
p1=tx*(0,3,0);
p2=tx*rotate(40,Z)*(0,3,0);
draw(p0--p1,cyan);
draw(p0--p2,cyan);

triple p1a=tx*(0,1.5,0);
draw("$\theta_{i}$",arc(p0,p1a,p2),ArcArrow3(3));

// draw d_{i-1}
triple org_i   =t2*shift((0,0,1.5))*O;
draw(Label("$d_{i}$",0.13),p0--org_i,linewidth(1pt)); 

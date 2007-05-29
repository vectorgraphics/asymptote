int ncell=10;

import graph_settings;
import light;

real eps=10000*realEpsilon;

private struct weighted
{
  triple normal;
  real ratio;
  int kpa0,kpa1,kpa2;
  int kpb0,kpb1,kpb2;
  triple pt;
}

private struct bucket
{
  triple t;
  triple val;
  int count;
  pair z;
}

struct vertex
{
  pair z;
  triple normal;
}

// A group of 3 or 4 points.
private struct object
{
  bool active;
  weighted[] pts;
}

private weighted setupweighted(triple va, triple vb, real da, real db, 
                               int kpa0, int kpa1, int kpa2,
                               int kpb0, int kpb1, int kpb2)
{
  weighted newone;
  real ratio=abs(da/(db-da));
  newone.pt=interp(va,vb,ratio);
  newone.ratio=ratio;
  newone.kpa0=kpa0;
  newone.kpa1=kpa1;
  newone.kpa2=kpa2;
  newone.kpb0=kpb0;
  newone.kpb1=kpb1;
  newone.kpb2=kpb2;

  return newone;
}

private weighted setupweighted(triple v, int kp0, int kp1, int kp2)
{
  weighted newone;
  newone.pt=v;
  newone.ratio=0.5;
  newone.kpa0=newone.kpb0=kp0;
  newone.kpa1=newone.kpb1=kp1;
  newone.kpa2=newone.kpb2=kp2;

  return newone;
}

// Checks if a pyramid contains a contour object.
private object checkpyr(triple v0, triple v1, triple v2, triple v3,
			real d0, real d1, real d2, real d3,
			int c00, int c01, int c02,
			int c10, int c11, int c12,
			int c20, int c21, int c22,
			int c30, int c31, int c32)
{
  object obj;
  real a0=abs(d0);
  real a1=abs(d1);
  real a2=abs(d2);
  real a3=abs(d3);

  bool b0=a0 < eps;
  bool b1=a1 < eps;
  bool b2=a2 < eps;
  bool b3=a3 < eps;
  bool s1=b0 || b1 ? false : abs(d0+d1)+eps < a0+a1;
  bool s2=b0 || b2 ? false : abs(d0+d2)+eps < a0+a2;
  bool s3=b0 || b3 ? false : abs(d0+d3)+eps < a0+a3;
  bool s4=b1 || b2 ? false : abs(d1+d2)+eps < a1+a2;
  bool s5=b1 || b3 ? false : abs(d1+d3)+eps < a1+a3;
  bool s6=b2 || b3 ? false : abs(d2+d3)+eps < a2+a3;

  weighted[] pts;
  if(b0) pts.push(setupweighted(v0,c00,c01,c02));
  if(b1) pts.push(setupweighted(v1,c10,c11,c12));
  if(b2) pts.push(setupweighted(v2,c20,c21,c22));
  if(b3) pts.push(setupweighted(v3,c30,c31,c32));
  if(s1) pts.push(setupweighted(v0,v1,d0,d1,c00,c01,c02,c10,c11,c12));
  if(s2) pts.push(setupweighted(v0,v2,d0,d2,c00,c01,c02,c20,c21,c22));
  if(s3) pts.push(setupweighted(v0,v3,d0,d3,c00,c01,c02,c30,c31,c32));
  if(s4) pts.push(setupweighted(v1,v2,d1,d2,c10,c11,c12,c20,c21,c22));
  if(s5) pts.push(setupweighted(v1,v3,d1,d3,c10,c11,c12,c30,c31,c32));
  if(s6) pts.push(setupweighted(v2,v3,d2,d3,c20,c21,c22,c30,c31,c32));

  int s=pts.length;
  //There are three or four points.
  if(s > 2) {
    obj.active=true;
    obj.pts=pts;
  }
  else obj.active=false;

  return obj;
}

// Return contour vertices for a 3D data array, using a pyramid mesh
// f,mp:      three-dimensional arrays of real data values
// a,b:       'bottom' and 'top' vertices of contour domain
vertex[][] contour3(real[][][] f, real[][][] mp=new real[][][] ,
                    triple a, triple b,
                    projection P=currentprojection)
{
  int nx=f.length-1;
  if(nx == 0)
    abort("array f must have length >= 2");
  int ny=f[0].length-1;
  if(ny == 0)
    abort("array f[0] must have length >= 2");
  int nz=f[0][0].length-1;
  if(nz == 0)
    abort("array f[0][0] must have length >= 2");
 
  // check if boundaries are good
  if(b.x <= a.x || b.y <= a.y) {
    abort("bad contour domain: all coordinates of b-a must be positive.");
  }
 
  bool finite=!P.infinity;
  triple dir=P.camera;

  bool midpoints=mp.length > 0;
  if(!midpoints) mp=new real[2nx+1][2ny+1][2nz+1];

  bucket[][][][] kps=new bucket[2nx+1][2ny+1][2nz+1][];
  for(int i=0; i < 2nx+1; ++i)
    for(int j=0; j < 2ny+1; ++j)
      for(int k=0; k < 2nz+1; ++k)
        kps[i][j][k]=new bucket[];

  for(int i=0; i <= nx; ++i)
    for(int j=0; j <= ny; ++j)
      for(int k=0; k <= nz; ++k)
        mp[2i][2j][2k]=f[i][j][k];   

  object[] objects;

  real dx=(b.x-a.x)/nx;
  real dy=(b.y-a.y)/ny;
  real dz=(b.z-a.z)/nz;
  
  // go over region a rectangle at a time
  for(int i=0; i < nx; ++i) {
    real x=a.x+i*dx;
    real[][] fi=f[i];
    real[][] fi1=f[i+1];
    int i2=2i;
    int i2p1=i2+1;
    int i2p2=i2+2;
    for(int j=0; j < ny; ++j) {
      real y=a.y+j*dy;
      real[] fij=fi[j];
      real[] fi1j=fi1[j];
      real[] fij1=fi[j+1];
      real[] fi1j1=fi1[j+1];
      int j2=2j;
      int j2p1=j2+1;
      int j2p2=j2+2;
 
      for(int k=0; k < nz; ++k) {
        real z=a.z+k*dz;

        // vertex values
        real vdat0=fij[k];
        real vdat1=fij[k+1];
        real vdat2=fij1[k];
        real vdat3=fij1[k+1];
        real vdat4=fi1j[k];
        real vdat5=fi1j[k+1];
        real vdat6=fi1j1[k];
        real vdat7=fi1j1[k+1];

        // define points
        triple p000=(x,y,z);
        triple p001=(x,y,z+dz);
        triple p010=(x,y+dy,z);
        triple p011=(x,y+dy,z+dz);
        triple p100=(x+dx,y,z);
        triple p101=(x+dx,y,z+dz);
        triple p110=(x+dx,y+dy,z);
        triple p111=(x+dx,y+dy,z+dz);
        triple m0=0.25*(p000+p010+p110+p100);
        triple m1=0.25*(p010+p110+p111+p011);
        triple m2=0.25*(p110+p100+p101+p111);
        triple m3=0.25*(p100+p000+p001+p101);
        triple m4=0.25*(p000+p010+p011+p001);
        triple m5=0.25*(p001+p011+p111+p101);
        triple mc=0.5*(m0+m5);                   

        // optimization: we make sure we don't work with empty rectangles
        int countm=0;
        int countz=0;
        int countp=0;
        
        void check(real vdat) {
          if(vdat < -eps) ++countm;
          else {
            if(vdat <= eps) ++countz; 
            else ++countp;
          }
        }
        
        check(vdat0);
        check(vdat1);
        check(vdat2);
        check(vdat3);
        check(vdat4);
        check(vdat5);
        check(vdat6);
        check(vdat7);

        if(countm == 8 || countp == 8 || 
           ((countm == 7 || countp == 7) && countz == 1)) continue;

        int k2=2k;
        int k2p1=k2+1;
        int k2p2=k2+2;
 
        // Evaluate midpoints of cube sides.
        // Then evaluate midpoint of cube.
        real vdat8=midpoints ? mp[i2p1][j2p1][k2] :
          0.25*(vdat0+vdat2+vdat6+vdat4);
        real vdat9=midpoints ? mp[i2p1][j2p2][k2p1] : 
          0.25*(vdat2+vdat6+vdat7+vdat3);
        real vdat10=midpoints ? mp[i2p2][j2p1][k2p1] : 
          0.25*(vdat7+vdat6+vdat4+vdat5);
        real vdat11=midpoints ? mp[i2p1][j2][k2p1] : 
          0.25*(vdat0+vdat4+vdat5+vdat1);
        real vdat12=midpoints ? mp[i2][j2p1][k2p1] : 
          0.25*(vdat0+vdat2+vdat3+vdat1);
        real vdat13=midpoints ? mp[i2p1][j2p1][k2p2] : 
          0.25*(vdat1+vdat3+vdat7+vdat5);
        real vdat14=midpoints ? mp[i2p1][j2p1][k2p1] : 
          0.125*(vdat0+vdat1+vdat2+vdat3+vdat4+vdat5+vdat6+vdat7);
      
        // Go through the 24 pyramids, 4 for each side.
        
        void addval(int kp0, int kp1, int kp2, triple add, triple pt) {
          bucket[] cur=kps[kp0][kp1][kp2];
          for(int q=0; q < cur.length; ++q) {
            if(length(cur[q].t-pt) < eps) {
              cur[q].val += add;
              ++cur[q].count;
              return;
            }
          }
          bucket newbuck;
          newbuck.t=pt;
          newbuck.z=project(pt,P);
          newbuck.val=add;
          newbuck.count=1;
          cur.push(newbuck);
        }

        void accrue(weighted w) {
          triple val1=w.normal*w.ratio;
          triple val2=w.normal*(1-w.ratio);
          addval(w.kpa0,w.kpa1,w.kpa2,val1,w.pt);
          addval(w.kpb0,w.kpb1,w.kpb2,val2,w.pt);
        }

        void addnormals(weighted[] pts) {
          triple normal0=cross(pts[1].pt-pts[0].pt,pts[2].pt-pts[0].pt);
          triple normal1=cross(pts[2].pt-pts[1].pt,pts[0].pt-pts[1].pt);
          triple normal2=cross(pts[1].pt-pts[2].pt,pts[0].pt-pts[2].pt);
          if(finite) {
            normal0 *= sgn(dot(normal0,P.camera-normal0));
            normal1 *= sgn(dot(normal1,P.camera-normal1));
            normal2 *= sgn(dot(normal2,P.camera-normal2));
          } else {
            normal0 *= sgn(dot(normal0,dir));
            normal1 *= sgn(dot(normal1,dir));
            normal2 *= sgn(dot(normal2,dir));
          }
          pts[0].normal=normal0;
          pts[1].normal=normal1;
          pts[2].normal=normal2;
          return;
        }

        void addobj(object obj) {
          if(!obj.active) return;

          if(obj.pts.length == 4) {
            weighted[] points=obj.pts;
            object obj1;
            object obj2;
            obj1.active=true; 
            obj2.active=true;
            obj1.pts=new weighted[] {points[0],points[1],points[2]};
            obj2.pts=new weighted[] {points[1],points[2],points[3]};
            addobj(obj1);
            addobj(obj2);
          } else {
            addnormals(obj.pts);
            for(int q=0; q < obj.pts.length; ++q)
              accrue(obj.pts[q]);
            objects.push(obj);
          }
          return;
        }

        void check4pyr(triple v0, triple v1, triple v2, triple v3,
		       triple v4, triple v5,
		       real d0, real d1, real d2, real d3, real d4, real d5,
		       int c00, int c01, int c02,
		       int c10, int c11, int c12,
		       int c20, int c21, int c22,
		       int c30, int c31, int c32,
		       int c40, int c41, int c42,
		       int c50, int c51, int c52) {
          addobj(checkpyr(v5,v4,v0,v1,d5,d4,d0,d1,c50,c51,c52,
			  c40,c41,c42,c00,c01,c02,c10,c11,c12));
          addobj(checkpyr(v5,v4,v1,v2,d5,d4,d1,d2,c50,c51,c52,
			  c40,c41,c42,c10,c11,c12,c20,c21,c22));
          addobj(checkpyr(v5,v4,v2,v3,d5,d4,d2,d3,c50,c51,c52,
			  c40,c41,c42,c20,c21,c22,c30,c31,c32));
          addobj(checkpyr(v5,v4,v3,v0,d5,d4,d3,d0,c50,c51,c52,
			  c40,c41,c42,c30,c31,c32,c00,c01,c02));
        }

        check4pyr(p000,p010,p110,p100,mc,m0, 
                  vdat0,vdat2,vdat6,vdat4,vdat14,vdat8,
                  i2,j2,k2,
		  i2,j2p2,k2,
		  i2p2,j2p2,k2,
		  i2p2,j2,k2,
		  i2p1,j2p1,k2p1,
		  i2p1,j2p1,k2);
        check4pyr(p010,p110,p111,p011,mc,m1, 
                  vdat2,vdat6,vdat7,vdat3,vdat14,vdat9,
                  i2,j2p2,k2,
		  i2p2,j2p2,k2,
		  i2p2,j2p2,k2p2,
		  i2,j2p2,k2p2,
		  i2p1,j2p1,k2p1,
		  i2p1,j2p2,k2p1);
        check4pyr(p110,p100,p101,p111,mc,m2, 
                  vdat6,vdat4,vdat5,vdat7,vdat14,vdat10,
                  i2p2,j2p2,k2,
		  i2p2,j2,k2,
		  i2p2,j2,k2p2,
		  i2p2,j2p2,k2p2,
		  i2p1,j2p1,k2p1,
		  i2p2,j2p1,k2p1);
        check4pyr(p100,p000,p001,p101,mc,m3, 
                  vdat4,vdat0,vdat1,vdat5,vdat14,vdat11,
                  i2p2,j2,k2,
		  i2,j2,k2,
		  i2,j2,k2p2,
		  i2p2,j2,k2p2,
		  i2p1,j2p1,k2p1,
		  i2p1,j2,k2p1);
        check4pyr(p000,p010,p011,p001,mc,m4, 
                  vdat0,vdat2,vdat3,vdat1,vdat14,vdat12,
                  i2,j2,k2,
		  i2,j2p2,k2,
		  i2,j2p2,k2p2,
		  i2,j2,k2p2,
		  i2p1,j2p1,k2p1,
		  i2,j2p1,k2p1);
        check4pyr(p001,p011,p111,p101,mc,m5, 
                  vdat1,vdat3,vdat7,vdat5,vdat14,vdat13,
                  i2,j2,k2p2,
		  i2,j2p2,k2p2,
		  i2p2,j2p2,k2p2,
		  i2p2,j2,k2p2,
		  i2p1,j2p1,k2p1,
		  i2p1,j2p1,k2p2);
      }
    }
  }

  vertex preparevertex(weighted w) {
    vertex ret;
    triple normal=O;
    bool first=true;
    bucket[] kp1=kps[w.kpa0][w.kpa1][w.kpa2];
    bucket[] kp2=kps[w.kpb0][w.kpb1][w.kpb2];
    bool found1=false;
    bool found2=false;
    int count=0;
    int stop=max(kp1.length,kp2.length);
    for(int r=0; r < stop; ++r) {
      if(!found1) {
        if(length(w.pt-kp1[r].t) < eps) {
          if(first) {
            ret.z=kp1[r].z;
            first=false;
          }
          normal += kp1[r].val;
          count += kp1[r].count;
          found1=true;
        }
      }
      if(!found2) {
        if(length(w.pt-kp2[r].t) < eps) {
          if(first) {
            ret.z=kp2[r].z;
            first=false;
          }
          normal += kp2[r].val;
          count += kp2[r].count;
          found2=true;
        }
      }
    }
    ret.normal=normal*2/count;
    return ret;
  }
  
  // Prepare return value.
  vertex[][] g;
  
  for(int q=0; q < objects.length; ++q) {
    object p=objects[q];
    g.push(new vertex[] {preparevertex(p.pts[0]),preparevertex(p.pts[1]),
          preparevertex(p.pts[2])});
  }
  return g;
}

// Return contour vertices for a 3D data array, using a pyramid mesh
// f:         Function from R3 to R.
// a,b:       'bottom' and 'top' vertices of contour domain
// nx,ny,nz   subdivisions on x, y and z axes
vertex[][] contour3(real f(real, real, real), triple a, triple b,
                    int nx=ncell, int ny=nx, int nz=nx,
                    projection P=currentprojection)
{
  // evaluate function at points and midpoints
  real[][][] dat=new real[nx+1][ny+1][nz+1];
  real[][][] midpoint=new real[2nx+2][2ny+2][2nz+1];

  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i/nx);
    real x2=interp(a.x,b.x,(i+0.5)/nx);
    real[][] dati=dat[i];
    real[][] midpointi2=midpoint[2i];
    real[][] midpointi2p1=midpoint[2i+1];
    for(int j=0; j <= ny; ++j) {
      real y=interp(a.y,b.y,j/ny);
      real y2=interp(a.y,b.y,(j+0.5)/ny);
      real datij[]=dati[j];
      real[] midpointi2p1j2=midpointi2p1[2j];
      real[] midpointi2p1j2p1=midpointi2p1[2j+1];
      real[] midpointi2j2p1=midpointi2[2j+1];
      for(int k=0; k <= nz; ++k) {
        real z=interp(a.z,b.z,k/nz);
        real z2=interp(a.z,b.z,(k+0.5)/nz);
        datij[k]=f(x,y,z);
	if(i == nx || j == ny || k == nz) continue;
        int k2p1=2k+1;
        midpointi2p1j2p1[2k]=f(x2,y2,z); 
        midpointi2p1j2p1[k2p1]=f(x2,y2,z2);
        midpointi2p1j2[k2p1]=f(x2,y,z2);
        midpointi2j2p1[k2p1]=f(x,y2,z2);
        if(i == 0) midpoint[2nx][2j+1][k2p1]=f(b.x,y2,z2);
        if(j == 0) midpointi2p1[2ny][k2p1]=f(x2,b.y,z2);
        if(k == 0) midpointi2p1j2p1[2nz]=f(x2,y2,b.z);
      }
    }
  }
  return contour3(dat,midpoint,a,b,P);
}

// Return contour guides for a 3D data array, using a pyramid mesh
void draw(picture pic=currentpicture, vertex[][] g, pen p=lightgray,
          light light=currentlight)
{
  begingroup(pic);
  int[] edges={0,0,0};
  for(int i=0; i < g.length; ++i) {
    vertex[] cur=g[i];
    pair p0=cur[0].z;
    pair p1=cur[1].z;
    pair p2=cur[2].z;
    pen pen0=light.intensity(cur[0].normal)*p;
    pen pen1=light.intensity(cur[1].normal)*p;
    pen pen2=light.intensity(cur[2].normal)*p;
    gouraudshade(pic,p0--p1--p2--cycle,new pen[] {pen0,pen1,pen2}, 
                 new pair[] {p0,p1,p2},edges);
  }
  endgroup(pic);
}

int ncell=10;

import graph_settings;
import light;

real eps=10000*realEpsilon;

private struct weighted
{
  triple normal;
  real ratio;
  int[] kp1=new int[3];
  int[] kp2=new int[3];
  triple pt;
}

private struct bucket
{
  triple t;
  triple val;
  int count;
}

private struct particle
{
  bool active;
  weighted[] pts;
}

private weighted setupweighted(triple va, triple vb, real da, real db, 
			       int[] kpa, int[] kpb)
{
  weighted newone;
  real ratio=abs(da/(db-da));
  newone.pt=interp(va,vb,ratio);
  newone.ratio=ratio;
  newone.kp1=kpa;
  newone.kp2=kpb;
  return newone;
}

private weighted setupweighted(triple v, int[] kp)
{
  weighted newone;
  newone.pt=v;
  newone.ratio=0.5;
  newone.kp1=kp;
  newone.kp2=kp;
  return newone;
}

// Checks if a pyramid contains a contour particle.
private particle checkpyr(triple[] v, real[] d, int[][] c)
{
  particle part;
  bool v0=(abs(d[0]) < eps);
  bool v1=(abs(d[1]) < eps);
  bool v2=(abs(d[2]) < eps);
  bool v3=(abs(d[3]) < eps);
  bool s1=(!v0 && !v1) ? (abs(d[0]+d[1])+eps < abs(d[0])+abs(d[1])) : false;
  bool s2=(!v0 && !v2) ? (abs(d[0]+d[2])+eps < abs(d[0])+abs(d[2])) : false;
  bool s3=(!v0 && !v3) ? (abs(d[0]+d[3])+eps < abs(d[0])+abs(d[3])) : false;
  bool s4=(!v1 && !v2) ? (abs(d[1]+d[2])+eps < abs(d[1])+abs(d[2])) : false;
  bool s5=(!v1 && !v3) ? (abs(d[1]+d[3])+eps < abs(d[1])+abs(d[3])) : false;
  bool s6=(!v2 && !v3) ? (abs(d[2]+d[3])+eps < abs(d[2])+abs(d[3])) : false;

  weighted[] pts;
  if(v0) pts.push(setupweighted(v[0],c[0]));
  if(v1) pts.push(setupweighted(v[1],c[1]));
  if(v2) pts.push(setupweighted(v[2],c[2]));
  if(v3) pts.push(setupweighted(v[3],c[3]));
  if(s1) pts.push(setupweighted(v[0],v[1],d[0],d[1],c[0],c[1]));
  if(s2) pts.push(setupweighted(v[0],v[2],d[0],d[2],c[0],c[2]));
  if(s3) pts.push(setupweighted(v[0],v[3],d[0],d[3],c[0],c[3]));
  if(s4) pts.push(setupweighted(v[1],v[2],d[1],d[2],c[1],c[2]));
  if(s5) pts.push(setupweighted(v[1],v[3],d[1],d[3],c[1],c[3]));
  if(s6) pts.push(setupweighted(v[2],v[3],d[2],d[3],c[2],c[3]));

  int s=pts.length;
  //There are three or four points.
  if(s > 2) {
    part.active=true;
    part.pts=pts;
  }
  else part.active=false;

  return part;
}

// Return contour buckets for a 3D data array, using a pyramid mesh
// f,mp:      three-dimensional arrays of real data values
// a,b:       'bottom' and 'top' vertices of contour domain
bucket[][] contour3(real[][][] f, real[][][] mp=new real[][][] ,
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

  particle[] particles;

  real dx=(b.x-a.x)/nx;
  real dy=(b.y-a.y)/ny;
  real dz=(b.z-a.z)/nz;
  
  // go over region a rectangle at a time
  for(int i=0; i < nx; ++i) {
    real x=a.x+i*dx;
    real[][] fi=f[i];
    real[][] fi1=f[i+1];
    for(int j=0; j < ny; ++j) {
      real y=a.y+j*dy;
      real[] fij=fi[j];
      real[] fi1j=fi1[j];
      real[] fij1=fi[j+1];
      real[] fi1j1=fi1[j+1];
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
        triple mc=0.125*(p000+p001+p010+p011+p100+p101+p110+p111);

        int[] pp000=new int[] {2i,2j,2k};
        int[] pp001=new int[] {2i,2j,2k+2};
        int[] pp010=new int[] {2i,2j+2,2k};
        int[] pp011=new int[] {2i,2j+2,2k+2};
        int[] pp100=new int[] {2i+2,2j,2k};
        int[] pp101=new int[] {2i+2,2j,2k+2};
        int[] pp110=new int[] {2i+2,2j+2,2k};
        int[] pp111=new int[] {2i+2,2j+2,2k+2};
        int[] pm0=new int[] {2i+1,2j+1,2k};
        int[] pm1=new int[] {2i+1,2j+2,2k+1};
        int[] pm2=new int[] {2i+2,2j+1,2k+1};
        int[] pm3=new int[] {2i+1,2j,2k+1};
        int[] pm4=new int[] {2i,2j+1,2k+1};
        int[] pm5=new int[] {2i+1,2j+1,2k+2};
        int[] pmc=new int[] {2i+1,2j+1,2k+1};
 
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

	// Evaluate midpoints of cube sides.
	// Then evaluate midpoint of cube.
	real vdat8=midpoints ? mp[2i+1][2j+1][2k] :
	  0.25*(vdat0+vdat2+vdat6+vdat4);
	real vdat9=midpoints ? mp[2i+1][2j+2][2k+1] : 
	  0.25*(vdat2+vdat6+vdat7+vdat3);
	real vdat10=midpoints ? mp[2i+2][2j+1][2k+1] : 
	  0.25*(vdat7+vdat6+vdat4+vdat5);
	real vdat11=midpoints ? mp[2i+1][2j][2k+1] : 
	  0.25*(vdat0+vdat4+vdat5+vdat1);
	real vdat12=midpoints ? mp[2i][2j+1][2k+1] : 
	  0.25*(vdat0+vdat2+vdat3+vdat1);
	real vdat13=midpoints ? mp[2i+1][2j+1][2k+2] : 
	  0.25*(vdat1+vdat3+vdat7+vdat5);
	real vdat14=midpoints ? mp[2i+1][2j+1][2k+1] : 
	  0.125*(vdat0+vdat1+vdat2+vdat3+vdat4+vdat5+vdat6+vdat7);
      
	// Go through the 24 pyramids, 4 for each side.
        
	void addval(int[] kp, triple add, triple pt) {
	  bucket[] cur=kps[kp[0]][kp[1]][kp[2]];
	  for(int q=0; q < cur.length; ++q) {
	    if(length(cur[q].t-pt) < eps) {
	      cur[q].val += add;
	      ++cur[q].count;
	      return;
	    }
	  }
	  bucket newbuck;
	  newbuck.t=pt;
	  newbuck.val=add;
	  newbuck.count=1;
	  cur.push(newbuck);
	}

	void accrue(weighted w) {
	  triple val1=w.normal*w.ratio;
	  triple val2=w.normal*(1-w.ratio);
	  addval(w.kp1,val1,w.pt);
	  addval(w.kp2,val2,w.pt);
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

	void addpart(particle part) {
	  if(!part.active) return;

	  if(part.pts.length == 4) {
	    weighted[] points=part.pts;
	    particle part1;
	    particle part2;
	    part1.active=true; 
	    part2.active=true;
	    part1.pts=new weighted[] {points[0],points[1],points[2]};
	    part2.pts=new weighted[] {points[1],points[2],points[3]};
	    addpart(part1);
	    addpart(part2);
	  } else {
	    addnormals(part.pts);
	    for(int q=0; q < part.pts.length; ++q)
	      accrue(part.pts[q]);
	    particles.push(part);
	  }
	  return;
	}

	void check4pyr(triple[] v, real[] d, int[][] c) {
	  addpart(checkpyr(new triple[] {v[5],v[4],v[0],v[1]},
			   new real[] {d[5],d[4],d[0],d[1]},
			   new int[][] {c[5],c[4],c[0],c[1]}));
	  addpart(checkpyr(new triple[] {v[5],v[4],v[1],v[2]},
			   new real[] {d[5],d[4],d[1],d[2]},
			   new int[][] {c[5],c[4],c[1],c[2]}));
	  addpart(checkpyr(new triple[] {v[5],v[4],v[2],v[3]},
			   new real[] {d[5],d[4],d[2],d[3]},
			   new int[][] {c[5],c[4],c[2],c[3]}));
	  addpart(checkpyr(new triple[] {v[5],v[4],v[3],v[0]},
			   new real[] {d[5],d[4],d[3],d[0]},
			   new int[][] {c[5],c[4],c[3],c[0]}));
	}

	check4pyr(new triple[] {p000,p010,p110,p100,mc,m0}, 
		  new real[] {vdat0,vdat2,vdat6,vdat4,vdat14,vdat8},
		  new int[][] {pp000,pp010,pp110,pp100,pmc,pm0});
	check4pyr(new triple[] {p010,p110,p111,p011,mc,m1}, 
		  new real[] {vdat2,vdat6,vdat7,vdat3,vdat14,vdat9},
		  new int[][] {pp010,pp110,pp111,pp011,pmc,pm1});
	check4pyr(new triple[] {p110,p100,p101,p111,mc,m2}, 
		  new real[] {vdat6,vdat4,vdat5,vdat7,vdat14,vdat10},
		  new int[][] {pp110,pp100,pp101,pp111,pmc,pm2});
	check4pyr(new triple[] {p100,p000,p001,p101,mc,m3}, 
		  new real[] {vdat4,vdat0,vdat1,vdat5,vdat14,vdat11},
		  new int[][] {pp100,pp000,pp001,pp101,pmc,pm3});
	check4pyr(new triple[] {p000,p010,p011,p001,mc,m4}, 
		  new real[] {vdat0,vdat2,vdat3,vdat1,vdat14,vdat12},
		  new int[][] {pp000,pp010,pp011,pp001,pmc,pm4});
	check4pyr(new triple[] {p001,p011,p111,p101,mc,m5}, 
		  new real[] {vdat1,vdat3,vdat7,vdat5,vdat14,vdat13},
		  new int[][] {pp001,pp011,pp111,pp101,pmc,pm5});
      }
    }
  }

  bucket preparebucket(weighted w) {
    bucket ret;
    ret.t=w.pt;
    ret.val=O;
    bucket[] kp1=kps[w.kp1[0]][w.kp1[1]][w.kp1[2]];
    bucket[] kp2=kps[w.kp2[0]][w.kp2[1]][w.kp2[2]];
    bool found1=false;
    bool found2=false;
    int count=0;
    int stop=max(kp1.length,kp2.length);
    for(int r=0; r < stop; ++r) {
      if(!found1) {
        if(length(w.pt-kp1[r].t) < eps) {
          ret.val += kp1[r].val;
          count += kp1[r].count;
          found1=true;
	}
      }
      if(!found2) {
        if(length(w.pt-kp2[r].t) < eps) {
          ret.val += kp2[r].val;
          count += kp2[r].count;
          found2=true;
	}
      }
    }
    ret.val *= 2/count;
    return ret;
  }
  
  // Prepare return value.
  bucket[][] g;
  
  for(int q=0; q < particles.length; ++q) {
    particle p=particles[q];
    g.push(new bucket[] {preparebucket(p.pts[0]),preparebucket(p.pts[1]),
	  preparebucket(p.pts[2])});
  }
  return g;
}

// Return contour buckets for a 3D data array, using a pyramid mesh
// f:         Function from R3 to R.
// a,b:       'bottom' and 'top' vertices of contour domain
// nx,ny,nz   subdivisions on x, y and z axes
bucket[][] contour3(real f(real, real, real), triple a, triple b,
		    int nx=ncell, int ny=nx, int nz=nx,
		    projection P=currentprojection)
{
  // evaluate function at points and midpoints
  real[][][] dat=new real[nx+1][ny+1][nz+1];
  real[][][] midpoint=new real[2nx+1][2ny+1][2nz+1];

  for(int i=0; i <= nx; ++i) {
    real x=interp(a.x,b.x,i/nx);
    real x2=interp(a.x,b.x,(i+0.5)/nx);
    real[][] dati=dat[i];
    for(int j=0; j <= ny; ++j) {
      real y=interp(a.y,b.y,j/ny);
      real y2=interp(a.y,b.y,(j+0.5)/ny);
      real datij[]=dati[j];
      for(int k=0; k <= nz; ++k) {
        real z=interp(a.z,b.z,k/nz);
        real z2=interp(a.z,b.z,(k+0.5)/nz);
        datij[k]=f(x,y,z);
        if(i == nx || j == ny || k == nz) continue;
        midpoint[2i+1][2j+1][2k]=f(x2,y2,z); 
        midpoint[2i+1][2j][2k+1]=f(x2,y,z2);
        midpoint[2i][2j+1][2k+1]=f(x,y2,z2);
        midpoint[2i+1][2j+1][2k+1]=f(x2,y2,z2);
        if(i == 0) midpoint[2nx][2j+1][2k+1]=f(b.x,y2,z2);
        if(j == 0) midpoint[2i+1][2ny][2k+1]=f(x2,b.y,z2);
        if(k == 0) midpoint[2i+1][2j+1][2nz]=f(x2,y2,b.z);
      }
    }
  }
  return contour3(dat,midpoint,a,b,P);
}

// Return contour guides for a 3D data array, using a pyramid mesh
void draw(picture pic=currentpicture, bucket[][] g, pen p=lightgray,
	  light light=currentlight, projection P=currentprojection)
{
  begingroup(pic);
  int[] edges={0,0,0};
  for(int i=0; i < g.length; ++i) {
    bucket[] cur=g[i];
    pair p0=project(cur[0].t,P);
    pair p1=project(cur[1].t,P);
    pair p2=project(cur[2].t,P);
    pen pen0=light.intensity(cur[0].val)*p;
    pen pen1=light.intensity(cur[1].val)*p;
    pen pen2=light.intensity(cur[2].val)*p;
    gouraudshade(pic,p0--p1--p2--cycle,new pen[] {pen0,pen1,pen2}, 
		 new pair[] {p0,p1,p2},edges);
  }
  endgroup(pic);
}

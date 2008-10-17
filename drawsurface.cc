/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"

namespace camp {

const double pixel=1.0; // Adaptive rendering constant.

using vm::array;
triple drawSurface::c3[];

inline void initMatrix(GLfloat *v, double x, double ymin, double zmin,
		       double ymax, double zmax)
{
  v[0]=x;
  v[1]=ymin;
  v[2]=zmin;
  v[3]=1.0;
  v[4]=x;
  v[5]=ymin;
  v[6]=zmax;
  v[7]=1.0;
  v[8]=x;
  v[9]=ymax;
  v[10]=zmin;
  v[11]=1.0;
  v[12]=x;
  v[13]=ymax;
  v[14]=zmax;
  v[15]=1.0;
}

void drawSurface::bounds(bbox3& b)
{
  static double c1[16];

  for(int i=0; i < 16; ++i)
    c1[i]=controls[i][0];
  double c0=c1[0];
  double xmin=bound(c1,min,b.empty ? c0 : min(c0,b.left));
  double xmax=bound(c1,max,b.empty ? c0 : max(c0,b.right));
    
  for(int i=0; i < 16; ++i)
    c1[i]=controls[i][1];
  c0=c1[0];
  double ymin=bound(c1,min,b.empty ? c0 : min(c0,b.bottom));
  double ymax=bound(c1,max,b.empty ? c0 : max(c0,b.top));
    
  for(int i=0; i < 16; ++i)
    c1[i]=controls[i][2];
  c0=c1[0];
  double zmin=bound(c1,min,b.empty ? c0 : min(c0,b.lower));
  double zmax=bound(c1,max,b.empty ? c0 : max(c0,b.upper));
    
  Min=triple(xmin,ymin,zmin);
  Max=triple(xmax,ymax,zmax);
    
  b.add(Min);
  b.add(Max);
}

void drawSurface::bounds(pair &b, double (*m)(double, double),
			 double (*x)(const triple&, double*),
			 double (*y)(const triple&, double*),
			 double *t, bool &first)
{
  for(int i=0; i < 16; ++i) {
    double *ci=controls[i];
    c3[i]=triple(ci[0],ci[1],ci[2]);
  }
  
  if(first) {
    triple v=c3[0];
    b=pair(x(v,t),y(v,t));
    first=false;
  }
  
  b=pair(bound(c3,m,x,t,b.getx()),bound(c3,m,y,t,b.gety()));
}

bool drawSurface::write(prcfile *out)
{
  if(invisible)
    return true;

  PRCMaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  out->add(new PRCBezierSurface(out,3,3,4,4,controls,m,granularity));
  
  return true;
}

// return the perpendicular displacement of a point z from the plane
// through u with unit normal n.
inline triple displacement2(const Triple& z, const Triple& u, const triple& n)
{
  triple Z=triple(z[0]-u[0],z[1]-u[1],z[2]-u[2]);
  return n != triple(0,0,0) ? dot(Z,n)*n : Z;
}

inline triple maxabs(triple u, triple v)
{
  return triple(max(fabs(u.getx()),fabs(v.getx())),
		max(fabs(u.gety()),fabs(v.gety())),
		max(fabs(u.getz()),fabs(v.getz())));
}

inline triple displacement(const Triple& z0, const Triple& c0,
			   const Triple& c1, const Triple& z1)
{
  triple Z0(z0[0],z0[1],z0[2]);
  triple Z1(z1[0],z1[1],z1[2]);
  return maxabs(displacement(triple(c0[0],c0[1],c0[2]),Z0,Z1),
		displacement(triple(c1[0],c1[1],c1[2]),Z0,Z1));
}

void drawSurface::displacement()
{
#ifdef HAVE_LIBGLUT
  initMatrix(v1,Min.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  initMatrix(v2,Max.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  
  for(int i=0; i < 16; ++i)
    store(c+3*i,controls[i]);

  static const triple zero;
  havenormal=normal != zero;
  if(havenormal) {
    store(Normal,normal);
    d=zero;
    
    if(!straight) {
      for(int i=1; i < 16; ++i) 
	d=camp::maxabs(d,camp::displacement2(controls[i],controls[0],normal));
      
      dperp=d;
    
      for(int i=0; i < 4; ++i)
	d=camp::maxabs(d,camp::displacement(controls[4*i],controls[4*i+1],
					    controls[4*i+2],controls[4*i+3]));
      for(int i=0; i < 4; ++i)
	d=camp::maxabs(d,camp::displacement(controls[i],controls[i+4],
					    controls[i+8],controls[i+12]));
    }
  }
#endif  
}
  
inline double fraction(double d, double size)
{
  return size == 0 ? 1.0 : min(fabs(d)/size,1.0);
}

// estimate the viewport fraction associated with the displacement d
inline double fraction(const triple& d, const triple& size)
{
  return max(max(fraction(d.getx(),size.getx()),
		 fraction(d.gety(),size.gety())),
	     fraction(d.getz(),size.getz()));
}

void drawSurface::render(GLUnurbs *nurb, double size2,
			 const triple& Min, const triple& Max,
			 double perspective, bool transparent, bool twosided)
{
#ifdef HAVE_LIBGLUT
  if(invisible || ((diffuse.A < 1.0) ^ transparent)) return;
  
  static GLfloat v[16];

  glPushMatrix();
  glMultMatrixf(v1);
  glGetFloatv(GL_MODELVIEW_MATRIX,v);
  glPopMatrix();
  
  bbox3 B(v[0],v[1],v[2]);
  B.addnonempty(v[4],v[5],v[6]);
  B.addnonempty(v[8],v[9],v[10]);
  B.addnonempty(v[12],v[13],v[14]);
  
  glPushMatrix();
  glMultMatrixf(v2);
  glGetFloatv(GL_MODELVIEW_MATRIX,v);
  glPopMatrix();
  
  B.addnonempty(v[0],v[1],v[2]);
  B.addnonempty(v[4],v[5],v[6]);
  B.addnonempty(v[8],v[9],v[10]);
  B.addnonempty(v[12],v[13],v[14]);
  
  triple M=B.Max();
  triple m=B.Min();
  
  if(perspective) {
    double f=m.getz()*perspective;
    double F=M.getz()*perspective;
    if(M.getx() < min(f*Min.getx(),F*Min.getx()) || 
       m.getx() > max(f*Max.getx(),F*Max.getx()) ||
       M.gety() < min(f*Min.gety(),F*Min.gety()) ||
       m.gety() > max(f*Max.gety(),F*Max.gety()) ||
       M.getz() < Min.getz() ||
       m.getz() > Max.getz()) return;
  } else if(M.getx() < Min.getx() || m.getx() > Max.getx() ||
	    M.gety() < Min.gety() || m.gety() > Max.gety() ||
	    M.getz() < Min.getz() || m.getz() > Max.getz()) return;
    
  
  GLfloat Diffuse[]={diffuse.R,diffuse.G,diffuse.B,diffuse.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Diffuse);
  
  GLfloat Ambient[]={ambient.R,ambient.G,ambient.B,ambient.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Ambient);
  
  GLfloat Emissive[]={emissive.R,emissive.G,emissive.B,emissive.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Emissive);
  
  GLfloat Specular[]={specular.R,specular.G,specular.B,specular.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Specular);
  
  glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,128.0*shininess);

  triple size3=Max-Min;
  double f=fraction(d,size3);
  double fperp=fraction(dperp,size3);
    
  if(!havenormal || !straight && (f*size2 >= pixel || granularity == 0)) {
    if(havenormal && fperp*size2 <= 0.1) {
      glNormal3fv(Normal);
      gluNurbsCallback(nurb,GLU_NURBS_NORMAL,NULL);
    } else
      gluNurbsCallback(nurb,GLU_NURBS_NORMAL,(_GLUfuncptr) glNormal3fv);
    static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
    gluBeginSurface(nurb);
    gluNurbsSurface(nurb,8,knots,8,knots,3,12,c,4,4,GL_MAP2_VERTEX_3);
    gluEndSurface(nurb);
  } else {
    glBegin(GL_QUADS);
    glNormal3fv(Normal);
    glVertex3fv(c);
    glVertex3fv(c+9);
    glVertex3fv(c+45);
    glVertex3fv(c+36);
    glEnd();
  }
  
#endif
}

drawElement *drawSurface::transformed(const array& t)
{
  return new drawSurface(t,this);
}
  
} //namespace camp

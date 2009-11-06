/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"
#include "arrayop.h"

namespace camp {

const double pixel=1.0; // Adaptive rendering constant.
const triple drawSurface::zero;

using vm::array;

#ifdef HAVE_LIBGL
void storecolor(GLfloat *colors, int i, const vm::array &pens, int j)
{
  pen p=vm::read<camp::pen>(pens,j);
  p.torgb();
  colors[i]=p.red();
  colors[i+1]=p.green();
  colors[i+2]=p.blue();
  colors[i+3]=p.opacity();
}
#endif  

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
  double x,y,z;
  double X,Y,Z;
  
  if(straight) {
    double *v=vertices[0];
    x=v[0];
    y=v[1];
    z=v[2];
    X=x;
    Y=y;
    Z=z;
    for(size_t i=1; i < 4; ++i) {
      double *v=vertices[i];
      double vx=v[0];
      x=min(x,vx);
      X=max(X,vx);
      double vy=v[1];
      y=min(y,vy);
      Y=max(Y,vy);
      double vz=v[2];
      z=min(z,vz);
      Z=max(Z,vz);
    }
  } else {
    static double c1[16];

    for(int i=0; i < 16; ++i)
      c1[i]=controls[i][0];
    double c0=c1[0];
    double fuzz=sqrtFuzz*run::norm(c1,16);
    x=bound(c1,min,b.empty ? c0 : min(c0,b.left),fuzz);
    X=bound(c1,max,b.empty ? c0 : max(c0,b.right),fuzz);
    
    for(int i=0; i < 16; ++i)
      c1[i]=controls[i][1];
    c0=c1[0];
    fuzz=sqrtFuzz*run::norm(c1,16);
    y=bound(c1,min,b.empty ? c0 : min(c0,b.bottom),fuzz);
    Y=bound(c1,max,b.empty ? c0 : max(c0,b.top),fuzz);
    
    for(int i=0; i < 16; ++i)
      c1[i]=controls[i][2];
    c0=c1[0];
    fuzz=sqrtFuzz*run::norm(c1,16);
    z=bound(c1,min,b.empty ? c0 : min(c0,b.lower),fuzz);
    Z=bound(c1,max,b.empty ? c0 : max(c0,b.upper),fuzz);
  }
    
  Min=triple(x,y,z);
  Max=triple(X,Y,Z);
    
  b.add(Min);
  b.add(Max);
}

void drawSurface::ratio(pair &b, double (*m)(double, double), bool &first)
{
  if(straight) {
    if(first) {
      first=false;
      double *ci=vertices[0];
      triple v=triple(ci[0],ci[1],ci[2]);
      b=pair(xratio(v),yratio(v));
    }
  
    double x=b.getx();
    double y=b.gety();
    for(size_t i=0; i < 4; ++i) {
      double *ci=vertices[i];
      triple v=triple(ci[0],ci[1],ci[2]);
      x=m(x,xratio(v));
      y=m(y,yratio(v));
    }
    b=pair(x,y);
  } else {
    static triple c3[16];
    for(int i=0; i < 16; ++i) {
      double *ci=controls[i];
      c3[i]=triple(ci[0],ci[1],ci[2]);
    }
  
    if(first) {
      triple v=c3[0];
      b=pair(xratio(v),yratio(v));
      first=false;
    }
  
    double fuzz=sqrtFuzz*run::norm(c3,16);
    b=pair(bound(c3,m,xratio,b.getx(),fuzz),bound(c3,m,yratio,b.gety(),fuzz));
  }
}

bool drawSurface::write(prcfile *out, unsigned int *count, array *index,
                        array *origin)
{
  if(invisible)
    return true;

  ostringstream buf;
  if(name == "") 
    buf << "surface-" << count[SURFACE]++;
  else
    buf << name;
  
  if(interaction == BILLBOARD) {
    size_t n=origin->size();
    
    if(n == 0 || center != vm::read<triple>(origin,n-1)) {
      origin->push(center);
      ++n;
    }
    
    unsigned int i=count[BILLBOARD_SURFACE]++;
    buf << "-" << i << "\001";
    index->push((Int) (n-1));
  }
  
  PRCMaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);

  if(straight)
    out->add(new PRCBezierSurface(out,1,1,2,2,vertices,m,granularity,
                                  buf.str()));
  else
    out->add(new PRCBezierSurface(out,3,3,4,4,controls,m,granularity,
                                  buf.str()));
  
  return true;
}

// return the perpendicular displacement of a point z from the plane
// through u with unit normal n.
inline triple displacement2(const Triple& z, const Triple& u, const triple& n)
{
  triple Z=triple(z)-triple(u);
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
  triple Z0(z0);
  triple Z1(z1);
  return maxabs(displacement(triple(c0[0],c0[1],c0[2]),Z0,Z1),
                displacement(triple(c1[0],c1[1],c1[2]),Z0,Z1));
}

void drawSurface::displacement()
{
#ifdef HAVE_LIBGL
  if(normal != zero) {
    d=zero;
    
    if(!straight) {
      for(size_t i=1; i < 16; ++i) 
        d=camp::maxabs(d,camp::displacement2(controls[i],controls[0],normal));
      
      dperp=d;
    
      for(size_t i=0; i < 4; ++i)
        d=camp::maxabs(d,camp::displacement(controls[4*i],controls[4*i+1],
                                            controls[4*i+2],controls[4*i+3]));
      for(size_t i=0; i < 4; ++i)
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

#ifdef HAVE_LIBGL
struct billboard 
{
  triple u,v,w;
  
  void init() {
    gl::projection P=gl::camera(false);
    w=unit(P.camera-P.target);
    v=unit(perp(P.up,w));
    u=cross(v,w);
  }
    
  void store(GLfloat* C, const triple& V,
             const triple &center=drawSurface::zero) {
    double cx=center.getx();
    double cy=center.gety();
    double cz=center.getz();
    double x=V.getx()-cx;
    double y=V.gety()-cy;
    double z=V.getz()-cz;
    C[0]=cx+u.getx()*x+v.getx()*y+w.getx()*z;
    C[1]=cy+u.gety()*x+v.gety()*y+w.gety()*z;
    C[2]=cz+u.getz()*x+v.getz()*y+w.getz()*z;
  }
};

billboard BB;
#endif

void drawSurface::render(GLUnurbs *nurb, double size2,
                         const triple& Min, const triple& Max,
                         double perspective, bool transparent)
{
#ifdef HAVE_LIBGL
  if(invisible || ((colors ? colors[3]+colors[7]+colors[11]+colors[15] < 4.0
                    : diffuse.A < 1.0) ^ transparent)) return;
  double s;
  static GLfloat Normal[3];

  static GLfloat v[16];
  static GLfloat v1[16];
  static GLfloat v2[16];
  
  initMatrix(v1,Min.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  initMatrix(v2,Max.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());

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
  
  bool havebillboard=interaction == BILLBOARD;
  
  if(perspective) {
    double f=m.getz()*perspective;
    double F=M.getz()*perspective;
    s=max(f,F);
    if(!havebillboard && (M.getx() < min(f*Min.getx(),F*Min.getx()) || 
                          m.getx() > max(f*Max.getx(),F*Max.getx()) ||
                          M.gety() < min(f*Min.gety(),F*Min.gety()) ||
                          m.gety() > max(f*Max.gety(),F*Max.gety()) ||
                          M.getz() < Min.getz() ||
                          m.getz() > Max.getz()))
      return;
  } else {
    s=1.0;
    if(!havebillboard && (M.getx() < Min.getx() || m.getx() > Max.getx() ||
                          M.gety() < Min.gety() || m.gety() > Max.gety() ||
                          M.getz() < Min.getz() || m.getz() > Max.getz()))
      return;
  }
    
  bool ambientdiffuse=true;
  bool emission=true;

  if(colors) {
    glEnable(GL_COLOR_MATERIAL);
    if(lighton) {
      glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
      ambientdiffuse=false;
    } else {
      glColorMaterial(GL_FRONT_AND_BACK,GL_EMISSION);
      emission=false;
    }
  }
  
  if(ambientdiffuse) {
    GLfloat Diffuse[]={diffuse.R,diffuse.G,diffuse.B,diffuse.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Diffuse);
  
    GLfloat Ambient[]={ambient.R,ambient.G,ambient.B,ambient.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Ambient);
  }
  
  if(emission) {
    GLfloat Emissive[]={emissive.R,emissive.G,emissive.B,emissive.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Emissive);
  }
  
  GLfloat Specular[]={specular.R,specular.G,specular.B,specular.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Specular);
  
  glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,128.0*shininess);

  triple size3=triple(s*(Max.getx()-Min.getx()),s*(Max.gety()-Min.gety()),
                      Max.getz()-Min.getz());
  
  bool havenormal=normal != zero;
  if(havebillboard) BB.init();

  if(!havenormal || (!straight && (fraction(d,size3)*size2 >= pixel || 
                                   granularity == 0))) {
    if(lighton) {
      if(havenormal && fraction(dperp,size3)*size2 <= 0.1) {
        if(havebillboard)
          BB.store(Normal,normal);
        else
          store(Normal,normal);
        glNormal3fv(Normal);
        gluNurbsCallback(nurb,GLU_NURBS_NORMAL,NULL);
      } else
        gluNurbsCallback(nurb,GLU_NURBS_NORMAL,(_GLUfuncptr) glNormal3fv);
    }
    static GLfloat Controls[48];
    
    if(havebillboard) {
      for(size_t i=0; i < 16; ++i)
        BB.store(Controls+3*i,controls[i],center);
    } else {
      for(size_t i=0; i < 16; ++i)
        store(Controls+3*i,controls[i]);
    }
    
    static GLfloat bezier[]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
    gluBeginSurface(nurb);
    gluNurbsSurface(nurb,8,bezier,8,bezier,12,3,Controls,4,4,GL_MAP2_VERTEX_3);
    if(colors) {
      static GLfloat linear[]={0.0,0.0,1.0,1.0};
      gluNurbsSurface(nurb,4,linear,4,linear,8,4,colors,2,2,GL_MAP2_COLOR_4);
    }
    
    gluEndSurface(nurb);
  } else {
    GLfloat Vertices[12];
    
    if(havebillboard) {
      for(size_t i=0; i < 4; ++i)
        BB.store(Vertices+3*i,vertices[i],center);
    } else {
      for(size_t i=0; i < 4; ++i)
        store(Vertices+3*i,vertices[i]);
    }
    
    if(havebillboard)
      BB.store(Normal,normal);
    else
      store(Normal,normal);

    glBegin(GL_QUADS);
    if(lighton)
      glNormal3fv(Normal);
    if(colors) 
      glColor4fv(colors);
    glVertex3fv(Vertices);
    if(colors) 
      glColor4fv(colors+8);
    glVertex3fv(Vertices+6);
    if(colors) 
      glColor4fv(colors+12);
    glVertex3fv(Vertices+9);
    if(colors) 
      glColor4fv(colors+4);
    glVertex3fv(Vertices+3);
    glEnd();
  }
  
  if(colors)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

drawElement *drawSurface::transformed(const array& t)
{
  return new drawSurface(t,this);
}
  
bool drawNurbs::write(prcfile *out, unsigned int *count, array *index,
                      array *origin)
{
  ostringstream buf;
  if(name == "") 
    buf << "surface-" << count[SURFACE]++;
  else
    buf << name;
  
  if(invisible)
    return true;

  PRCMaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);
  out->add(new PRCsurface(out,udegree,vdegree,nu,nv,controls,
                          uknots,vknots,m,scale3D,weights != NULL,
                          weights,granularity,name.c_str()));
  
  return true;
}

// Approximate bounds by bounding box of control polyhedron.
void drawNurbs::bounds(bbox3& b)
{
  size_t n=nu*nv;
  double *v=controls[0];
  double x=v[0];
  double y=v[1];
  double z=v[2];
  double X=x;
  double Y=y;
  double Z=z;
  for(size_t i=1; i < n; ++i) {
    double *v=controls[i];
    double vx=v[0];
    x=min(x,vx);
    X=max(X,vx);
    double vy=v[1];
    y=min(y,vy);
    Y=max(Y,vy);
    double vz=v[2];
    z=min(z,vz);
    Z=max(Z,vz);
  }

  Min=triple(x,y,z);
  Max=triple(X,Y,Z);
  b.add(Min);
  b.add(Max);
}

drawElement *drawNurbs::transformed(const array& t)
{
  return new drawNurbs(t,this);
}

void drawNurbs::ratio(pair &b, double (*m)(double, double), bool &first)
{
  size_t n=nu*nv;
  if(first) {
    first=false;
    double *ci=controls[0];
    triple v=triple(ci[0],ci[1],ci[2]);
    b=pair(xratio(v),yratio(v));
  }
  
  double x=b.getx();
  double y=b.gety();
  for(size_t i=0; i < n; ++i) {
    double *ci=controls[i];
    triple v=triple(ci[0],ci[1],ci[2]);
    x=m(x,xratio(v));
    y=m(y,yratio(v));
  }
  b=pair(x,y);
}

void drawNurbs::displacement()
{
#ifdef HAVE_LIBGL
  size_t n=nu*nv;
  size_t nuknots=udegree+nu+1;
  size_t nvknots=vdegree+nv+1;
    
  if(Controls == NULL) {
    Controls=new(UseGC)  GLfloat[(weights ? 4 : 3)*n];
    uKnots=new(UseGC) GLfloat[nuknots];
    vKnots=new(UseGC) GLfloat[nvknots];
  }
  
  if(weights)
    for(size_t i=0; i < n; ++i)
      store(Controls+4*i,controls[i],weights[i]);
  else
    for(size_t i=0; i < n; ++i)
      store(Controls+3*i,controls[i]);
  
  for(size_t i=0; i < nuknots; ++i)
    uKnots[i]=uknots[i];
  for(size_t i=0; i < nvknots; ++i)
    vKnots[i]=vknots[i];
#endif  
}

void drawNurbs::render(GLUnurbs *nurb, double size2,
                       const triple& Min, const triple& Max,
                       double perspective, bool transparent)
{
#ifdef HAVE_LIBGL
  if(invisible || ((colors ? colors[3]+colors[7]+colors[11]+colors[15] < 4.0
                    : diffuse.A < 1.0) ^ transparent)) return;
  
  static GLfloat v[16];
  static GLfloat v1[16];
  static GLfloat v2[16];

  initMatrix(v1,Min.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  initMatrix(v2,Max.getx(),Min.gety(),Min.getz(),Max.gety(),Max.getz());
  
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
  
  double s;
  if(perspective) {
    double f=m.getz()*perspective;
    double F=M.getz()*perspective;
    s=max(f,F);
    if(M.getx() < min(f*Min.getx(),F*Min.getx()) || 
       m.getx() > max(f*Max.getx(),F*Max.getx()) ||
       M.gety() < min(f*Min.gety(),F*Min.gety()) ||
       m.gety() > max(f*Max.gety(),F*Max.gety()) ||
       M.getz() < Min.getz() ||
       m.getz() > Max.getz()) return;
  } else {
    s=1.0;
    if(M.getx() < Min.getx() || m.getx() > Max.getx() ||
       M.gety() < Min.gety() || m.gety() > Max.gety() ||
       M.getz() < Min.getz() || m.getz() > Max.getz()) return;
  }

  bool ambientdiffuse=true;
  bool emission=true;

  if(colors) {
    glEnable(GL_COLOR_MATERIAL);
    if(lighton) {
      glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
      ambientdiffuse=false;
    } else {
      glColorMaterial(GL_FRONT_AND_BACK,GL_EMISSION);
      emission=false;
    }
  }
  
  if(ambientdiffuse) {
    GLfloat Diffuse[]={diffuse.R,diffuse.G,diffuse.B,diffuse.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Diffuse);
  
    GLfloat Ambient[]={ambient.R,ambient.G,ambient.B,ambient.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Ambient);
  }
  
  if(emission) {
    GLfloat Emissive[]={emissive.R,emissive.G,emissive.B,emissive.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Emissive);
  }
  
  GLfloat Specular[]={specular.R,specular.G,specular.B,specular.A};
  glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Specular);
  
  glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,128.0*shininess);

  gluNurbsCallback(nurb,GLU_NURBS_NORMAL,(_GLUfuncptr) glNormal3fv);
  gluBeginSurface(nurb);
  int uorder=udegree+1;
  int vorder=vdegree+1;
  size_t stride=weights ? 4 : 3;
  gluNurbsSurface(nurb,uorder+nu,uKnots,vorder+nv,vKnots,stride*nv,stride,
                  Controls,uorder,vorder,
                  weights ? GL_MAP2_VERTEX_4 : GL_MAP2_VERTEX_3);
  if(colors) {
    static GLfloat linear[]={0.0,0.0,1.0,1.0};
    gluNurbsSurface(nurb,4,linear,4,linear,8,4,colors,2,2,GL_MAP2_COLOR_4);
  }
    
  gluEndSurface(nurb);
  
  if(colors)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

} //namespace camp

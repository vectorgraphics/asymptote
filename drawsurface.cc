/*****
 * drawsurface.cc
 *
 * Stores a surface that has been added to a picture.
 *****/

#include "drawsurface.h"
#include "arrayop.h"

#include <iostream>
#include <iomanip>
#include <fstream>

namespace camp {

void bezierTriangle(const triple *g, double Size2, triple Size3,
                    bool havebillboard, triple center, GLfloat *colors);
  
const double pixel=1.0; // Adaptive rendering constant.
const triple drawElement::zero;

using vm::array;

#ifdef HAVE_GL
void storecolor(GLfloat *colors, int i, const vm::array &pens, int j)
{
  pen p=vm::read<camp::pen>(pens,j);
  p.torgb();
  colors[i]=p.red();
  colors[i+1]=p.green();
  colors[i+2]=p.blue();
  colors[i+3]=p.opacity();
}

void storecolor(GLfloat *colors, int i, const RGBAColour& p)
{
  colors[i]=p.R;
  colors[i+1]=p.G;
  colors[i+2]=p.B;
  colors[i+3]=p.A;
}

void setcolors(bool colors, bool lighton,
               const RGBAColour& diffuse,
               const RGBAColour& ambient,
               const RGBAColour& emissive,
               const RGBAColour& specular, double shininess) 
{
  if(colors) {
    glEnable(GL_COLOR_MATERIAL);
   if(!lighton) 
     glColorMaterial(GL_FRONT_AND_BACK,GL_EMISSION);

    GLfloat Black[]={0,0,0,(GLfloat) diffuse.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Black);
    glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Black);
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Black);
  } else {
    GLfloat Diffuse[]={(GLfloat) diffuse.R,(GLfloat) diffuse.G,
		       (GLfloat) diffuse.B,(GLfloat) diffuse.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Diffuse);
  
    GLfloat Ambient[]={(GLfloat) ambient.R,(GLfloat) ambient.G,
		       (GLfloat) ambient.B,(GLfloat) ambient.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Ambient);
  
    GLfloat Emissive[]={(GLfloat) emissive.R,(GLfloat) emissive.G,
			(GLfloat) emissive.B,(GLfloat) emissive.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_EMISSION,Emissive);
  }
    
  if(lighton) {
    GLfloat Specular[]={(GLfloat) specular.R,(GLfloat) specular.G,
			(GLfloat) specular.B,(GLfloat) specular.A};
    glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Specular);
  
    glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,128.0*shininess);
  }
}

#endif  

void drawSurface::bounds(const double* t, bbox3& b)
{
  double x,y,z;
  double X,Y,Z;
  
  if(straight) {
    triple *Vertices;
    if(t == NULL) Vertices=vertices;
    else {
      triple buf[4];
      Vertices=buf;
      for(int i=0; i < 4; ++i)
        Vertices[i]=t*vertices[i];
    }
  
    boundstriples(x,y,z,X,Y,Z,4,Vertices);
  } else {
    static double cx[16];
    static double cy[16];
    static double cz[16];
  
    if(t == NULL) {
      for(int i=0; i < 16; ++i) {
        triple v=controls[i];
        cx[i]=v.getx();
        cy[i]=v.gety();
        cz[i]=v.getz();
      }
    } else {
      for(int i=0; i < 16; ++i) {
        triple v=t*controls[i];
        cx[i]=v.getx();
        cy[i]=v.gety();
        cz[i]=v.getz();
      }
    }
    
    double c0=cx[0];
    double fuzz=sqrtFuzz*run::norm(cx,16);
    x=bound(cx,min,b.empty ? c0 : min(c0,b.left),fuzz,maxdepth);
    X=bound(cx,max,b.empty ? c0 : max(c0,b.right),fuzz,maxdepth);
    
    c0=cy[0];
    fuzz=sqrtFuzz*run::norm(cy,16);
    y=bound(cy,min,b.empty ? c0 : min(c0,b.bottom),fuzz,maxdepth);
    Y=boundtri(cy,max,b.empty ? c0 : max(c0,b.top),fuzz,maxdepth);
    
    c0=cz[0];
    fuzz=sqrtFuzz*run::norm(cz,16);
    z=bound(cz,min,b.empty ? c0 : min(c0,b.lower),fuzz,maxdepth);
    Z=bound(cz,max,b.empty ? c0 : max(c0,b.upper),fuzz,maxdepth);
  }  
  
  b.add(x,y,z);
  b.add(X,Y,Z);
  
  if(t == NULL) {
    Min=triple(x,y,z);
    Max=triple(X,Y,Z);
  }
}

void drawSurface::ratio(const double* t, pair &b, double (*m)(double, double),
                        double fuzz, bool &first)
{
  static triple buf[16];
  if(straight) {
    triple *Vertices;
    if(t == NULL) Vertices=vertices;
    else {
      Vertices=buf;
      for(int i=0; i < 4; ++i)
        Vertices[i]=t*vertices[i];
    }
  
    if(first) {
      first=false;
      triple v=Vertices[0];
      b=pair(xratio(v),yratio(v));
    }
  
    double x=b.getx();
    double y=b.gety();
    for(size_t i=0; i < 4; ++i) {
      triple v=Vertices[i];
      x=m(x,xratio(v));
      y=m(y,yratio(v));
    }
    b=pair(x,y);
  } else {
    triple* Controls;
    if(t == NULL) Controls=controls;
    else {
      Controls=buf;
      for(int i=0; i < 16; ++i)
        Controls[i]=t*controls[i];
    }

    if(first) {
      triple v=Controls[0];
      b=pair(xratio(v),yratio(v));
      first=false;
    }
  
    b=pair(bound(Controls,m,xratio,b.getx(),fuzz,maxdepth),
           bound(Controls,m,yratio,b.gety(),fuzz,maxdepth));
  }
}

bool drawSurface::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible || !prc)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);

  if(straight) {
    if(colors)
      out->addQuad(vertices,colors);
    else
      out->addRectangle(vertices,m);
  } else
    out->addPatch(controls,m);
                    
  return true;
}

// return the perpendicular displacement of a point z from the plane
// through u with unit normal n.
inline triple displacement2(const triple& z, const triple& u, const triple& n)
{
  triple Z=z-u;
  return n != triple(0,0,0) ? dot(Z,n)*n : Z;
}

inline triple maxabs(triple u, triple v)
{
  return triple(max(fabs(u.getx()),fabs(v.getx())),
                max(fabs(u.gety()),fabs(v.gety())),
                max(fabs(u.getz()),fabs(v.getz())));
}

inline triple displacement1(const triple& z0, const triple& c0,
                            const triple& c1, const triple& z1)
{
  return maxabs(displacement(c0,z0,z1),displacement(c1,z0,z1));
}

void drawSurface::displacement()
{
#ifdef HAVE_GL
  if(normal != zero) {
    d=zero;
    
    if(!straight) {
      for(size_t i=1; i < 16; ++i) 
        d=maxabs(d,displacement2(controls[i],controls[0],normal));
      
      dperp=d;
    
      for(size_t i=0; i < 4; ++i)
        d=maxabs(d,displacement1(controls[4*i],controls[4*i+1],
                                 controls[4*i+2],controls[4*i+3]));
      for(size_t i=0; i < 4; ++i)
        d=maxabs(d,displacement1(controls[i],controls[i+4],
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
                         double perspective, bool lighton, bool transparent)
{
#ifdef HAVE_GL
  if(invisible || 
     ((colors ? colors[0].A+colors[1].A+colors[2].A+colors[3].A < 4.0 :
       diffuse.A < 1.0) ^ transparent)) return;
  double s;
  static GLfloat Normal[3];

  static GLfloat v[16];
  
  const bool havebillboard=interaction == BILLBOARD &&
    !settings::getSetting<bool>("offscreen");
  triple m,M;
  if(perspective || !havebillboard) {
    static double t[16];
    glGetDoublev(GL_MODELVIEW_MATRIX,t);
// Like Fortran, OpenGL uses transposed (column-major) format!
    run::transpose(t,4);
    
    bbox3 B(this->Min,this->Max);
    B.transform(t);
  
    m=B.Min();
    M=B.Max();
  }

  if(perspective) {
    const double f=m.getz()*perspective;
    const double F=M.getz()*perspective;
    if(!havebillboard && (M.getx() < min(f*Min.getx(),F*Min.getx()) || 
                          m.getx() > max(f*Max.getx(),F*Max.getx()) ||
                          M.gety() < min(f*Min.gety(),F*Min.gety()) ||
                          m.gety() > max(f*Max.gety(),F*Max.gety()) ||
                          M.getz() < Min.getz() ||
                          m.getz() > Max.getz()))
      return;
    s=max(f,F);
  } else {
    if(!havebillboard && (M.getx() < Min.getx() || m.getx() > Max.getx() ||
                          M.gety() < Min.gety() || m.gety() > Max.gety() ||
                          M.getz() < Min.getz() || m.getz() > Max.getz()))
      return;
    s=1.0;
  }
    
  setcolors(colors,lighton,diffuse,ambient,emissive,specular,shininess);
  
  const triple size3(s*(Max.getx()-Min.getx()),s*(Max.gety()-Min.gety()),0.0);
  
  bool havenormal=normal != zero;
  if(havebillboard) BB.init();

  if(colors)
    for(size_t i=0; i < 4; ++i)
      storecolor(v,4*i,colors[i]);
    
  if(!havenormal || (!straight && fraction(d,size3)*size2 >= pixel)) {
    if(lighton) {
      if(havenormal && fraction(dperp,size3)*size2 <= 0.1) {
        if(havebillboard)
          BB.store(Normal,normal,zero);
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
      gluNurbsSurface(nurb,4,linear,4,linear,8,4,v,2,2,GL_MAP2_COLOR_4);
    }
    
    gluEndSurface(nurb);
  } else {
    static GLfloat Vertices[12];
    
    if(havebillboard) {
      for(size_t i=0; i < 4; ++i)
        BB.store(Vertices+3*i,vertices[i],center);
    } else {
      for(size_t i=0; i < 4; ++i)
        store(Vertices+3*i,vertices[i]);
    }
    
    if(havebillboard)
      BB.store(Normal,normal,zero);
    else
      store(Normal,normal);

    glBegin(GL_QUADS);
    if(lighton)
      glNormal3fv(Normal);
    if(colors) 
      glColor4fv(v);
    glVertex3fv(Vertices);
    if(colors) 
      glColor4fv(v+8);
    glVertex3fv(Vertices+6);
    if(colors) 
      glColor4fv(v+12);
    glVertex3fv(Vertices+9);
    if(colors) 
      glColor4fv(v+4);
    glVertex3fv(Vertices+3);
    glEnd();
  }
  
  if(colors)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

drawElement *drawSurface::transformed(const double* t)
{
  return new drawSurface(t,this);
}
  
void drawBezierTriangle::bounds(const double* t, bbox3& b)
{
  double x,y,z;
  double X,Y,Z;

  static double cx[10];
  static double cy[10];
  static double cz[10];
    
  if(t == NULL) {
    for(unsigned int i=0; i < 10; ++i) {
      triple v=controls[i];
      cx[i]=v.getx();
      cy[i]=v.gety();
      cz[i]=v.getz();
    }
  } else {
    for(unsigned int i=0; i < 10; ++i) {
      triple v=t*controls[i];
      cx[i]=v.getx();
      cy[i]=v.gety();
      cz[i]=v.getz();
    }
  }
    
  double c0=cx[0];
  double fuzz=sqrtFuzz*run::norm(cx,10);
  x=boundtri(cx,min,b.empty ? c0 : min(c0,b.left),fuzz,maxdepth);
  X=boundtri(cx,max,b.empty ? c0 : max(c0,b.right),fuzz,maxdepth);
    
  c0=cy[0];
  fuzz=sqrtFuzz*run::norm(cy,10);
  y=boundtri(cy,min,b.empty ? c0 : min(c0,b.bottom),fuzz,maxdepth);
  Y=boundtri(cy,max,b.empty ? c0 : max(c0,b.top),fuzz,maxdepth);
    
  c0=cz[0];
  fuzz=sqrtFuzz*run::norm(cz,10);
  z=boundtri(cz,min,b.empty ? c0 : min(c0,b.lower),fuzz,maxdepth);
  Z=boundtri(cz,max,b.empty ? c0 : max(c0,b.upper),fuzz,maxdepth);
    
  b.add(x,y,z);
  b.add(X,Y,Z);

  if(t == NULL) {
    Min=triple(x,y,z);
    Max=triple(X,Y,Z);
  }
}

void drawBezierTriangle::ratio(const double* t, pair &b,
                               double (*m)(double, double), double fuzz,
                               bool &first)
{
  triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    static triple buf[10];
    Controls=buf;
    for(unsigned int i=0; i < 10; ++i)
      Controls[i]=t*controls[i];
  }
    
  if(first) {
    triple v=Controls[0];
    b=pair(xratio(v),yratio(v));
    first=false;
  }
  
  b=pair(boundtri(Controls,m,xratio,b.getx(),fuzz,maxdepth),
         boundtri(Controls,m,yratio,b.gety(),fuzz,maxdepth));
}

bool drawBezierTriangle::write(prcfile *out, unsigned int *, double, 
                               groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);
  
  static const double third=1.0/3.0;
  static const double third2=2.0/3.0;
  triple Controls[]={controls[0],controls[0],controls[0],controls[0],
                       controls[1],third2*controls[1]+third*controls[2],
                       third*controls[1]+third2*controls[2],
                       controls[2],controls[3],
                       third*controls[3]+third2*controls[4],
                       third2*controls[4]+third*controls[5],
                       controls[5],controls[6],controls[7],
                       controls[8],controls[9]};
  out->addPatch(Controls,m);
                    
  return true;
}

void drawBezierTriangle::render(GLUnurbs *nurb, double size2,
                                const triple& Min, const triple& Max,
                                double perspective, bool lighton,
                                bool transparent)
{
#ifdef HAVE_GL
  if(invisible)
    return;

  if(invisible || 
     ((colors ? colors[0].A+colors[1].A+colors[2].A < 3.0 :
       diffuse.A < 1.0) ^ transparent)) return;

  double s;
  
  const bool havebillboard=interaction == BILLBOARD &&
    !settings::getSetting<bool>("offscreen");
  triple m,M;
  static double t[16]; // current transform
  glGetDoublev(GL_MODELVIEW_MATRIX,t);
// Like Fortran, OpenGL uses transposed (column-major) format!
  run::transpose(t,4);

  bbox3 B(this->Min,this->Max);
  B.transform(t);

  m=B.Min();
  M=B.Max();

  if(perspective) {
    const double f=m.getz()*perspective;
    const double F=M.getz()*perspective;
    if((M.getx() < min(f*Min.getx(),F*Min.getx()) ||
        m.getx() > max(f*Max.getx(),F*Max.getx()) ||
        M.gety() < min(f*Min.gety(),F*Min.gety()) ||
        m.gety() > max(f*Max.gety(),F*Max.gety()) ||
        M.getz() < Min.getz() ||
        m.getz() > Max.getz()))
      return;
    s=max(f,F);
  } else {
    if((M.getx() < Min.getx() || m.getx() > Max.getx() ||
        M.gety() < Min.gety() || m.gety() > Max.gety() ||
        M.getz() < Min.getz() || m.getz() > Max.getz()))
      return;
    s=1.0;
  }

  setcolors(colors,lighton,diffuse,ambient,emissive,specular,shininess);
  
  const triple size3(s*(Max.getx()-Min.getx()),s*(Max.gety()-Min.gety()),0);
  
  static GLfloat v[12];

  if(colors)
    for(size_t i=0; i < 3; ++i)
      storecolor(v,4*i,colors[i]);
    
  bezierTriangle(controls,size2,size3,havebillboard,center,colors ? v : NULL);

  if(colors)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

drawElement *drawBezierTriangle::transformed(const double* t)
{
  return new drawBezierTriangle(t,this);
}
  
bool drawNurbs::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);
  out->addSurface(udegree,vdegree,nu,nv,controls,uknots,vknots,m,weights);
  
  return true;
}

// Approximate bounds by bounding box of control polyhedron.
void drawNurbs::bounds(const double* t, bbox3& b)
{
  double x,y,z;
  double X,Y,Z;
  
  const size_t n=nu*nv;
  triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    Controls=new triple[n];
    for(size_t i=0; i < n; i++)
      Controls[i]=t*controls[i];
  }

  boundstriples(x,y,z,X,Y,Z,n,Controls);
  
  b.add(x,y,z);
  b.add(X,Y,Z);
  
  if(t == NULL) {
    Min=triple(x,y,z);
    Max=triple(X,Y,Z);
  } else delete[] Controls;
}

drawElement *drawNurbs::transformed(const double* t)
{
  return new drawNurbs(t,this);
}

void drawNurbs::ratio(const double *t, pair &b, double (*m)(double, double),
                      double, bool &first)
{
  const size_t n=nu*nv;
  
  triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    Controls=new triple[n];
    for(unsigned int i=0; i < n; ++i)
      Controls[i]=t*controls[i];
  }

  if(first) {
    first=false;
    triple v=Controls[0];
    b=pair(xratio(v),yratio(v));
  }
  
  double x=b.getx();
  double y=b.gety();
  for(size_t i=0; i < n; ++i) {
    triple v=Controls[i];
    x=m(x,xratio(v));
    y=m(y,yratio(v));
  }
  b=pair(x,y);
  
  if(t != NULL)
    delete[] Controls;
}


void drawNurbs::displacement()
{
#ifdef HAVE_GL
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
                       double perspective, bool lighton, bool transparent)
{
#ifdef HAVE_GL
  if(invisible || ((colors ? colors[3]+colors[7]+colors[11]+colors[15] < 4.0
                    : diffuse.A < 1.0) ^ transparent)) return;
  
  static double t[16]; // current transform
  glGetDoublev(GL_MODELVIEW_MATRIX,t);
  run::transpose(t,4);

  bbox3 B(this->Min,this->Max);
  B.transform(t);
    
  triple m=B.Min();
  triple M=B.Max();
  
  if(perspective) {
    double f=m.getz()*perspective;
    double F=M.getz()*perspective;
    if(M.getx() < min(f*Min.getx(),F*Min.getx()) || 
       m.getx() > max(f*Max.getx(),F*Max.getx()) ||
       M.gety() < min(f*Min.gety(),F*Min.gety()) ||
       m.gety() > max(f*Max.gety(),F*Max.gety()) ||
       M.getz() < Min.getz() ||
       m.getz() > Max.getz()) return;
  } else {
    if(M.getx() < Min.getx() || m.getx() > Max.getx() ||
       M.gety() < Min.gety() || m.gety() > Max.gety() ||
       M.getz() < Min.getz() || m.getz() > Max.getz()) return;
  }

  setcolors(colors,lighton,diffuse,ambient,emissive,specular,shininess);
  
  gluBeginSurface(nurb);
  int uorder=udegree+1;
  int vorder=vdegree+1;
  size_t stride=weights ? 4 : 3;
  gluNurbsSurface(nurb,uorder+nu,uKnots,vorder+nv,vKnots,stride*nv,stride,
                  Controls,uorder,vorder,
                  weights ? GL_MAP2_VERTEX_4 : GL_MAP2_VERTEX_3);
  if(lighton)
    gluNurbsCallback(nurb,GLU_NURBS_NORMAL,(_GLUfuncptr) glNormal3fv);
  
  if(colors) {
    static GLfloat linear[]={0.0,0.0,1.0,1.0};
    gluNurbsSurface(nurb,4,linear,4,linear,8,4,colors,2,2,GL_MAP2_COLOR_4);
  }
    
  gluEndSurface(nurb);
  
  if(colors)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

void drawSphere::P(triple& t, double x, double y, double z)
{
  if(half) {
    double temp=z; z=x; x=-temp;
  }
  
  if(T == NULL) {
    t=triple(x,y,z);
    return;
  }
  
  double f=T[12]*x+T[13]*y+T[14]*z+T[15];
  if(f == 0.0) run::dividebyzero();
  f=1.0/f;
  
  t=triple((T[0]*x+T[1]*y+T[2]*z+T[3])*f,(T[4]*x+T[5]*y+T[6]*z+T[7])*f,
           (T[8]*x+T[9]*y+T[10]*z+T[11])*f);
}

bool drawSphere::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  
  switch(type) {
    case 0: // PRCsphere
    {
      if(half) 
        out->addHemisphere(1.0,m,NULL,NULL,NULL,1.0,T);
      else
        out->addSphere(1.0,m,NULL,NULL,NULL,1.0,T);
      break;
    }
    case 1: // NURBSsphere
    {
      static double uknot[]={0.0,0.0,1.0/3.0,0.5,1.0,1.0};
      static double vknot[]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
      static double Weights[12]={2.0/3.0,2.0/9.0,2.0/9.0,2.0/3.0,
                                 1.0/3.0,1.0/9.0,1.0/9.0,1.0/3.0,
                                 1.0,1.0/3.0,1.0/3.0,1.0};

// NURBS representation of a sphere using 10 distinct control points
// K. Qin, J. Comp. Sci. and Tech. 12, 210-216 (1997).
  
      triple N,S,P1,P2,P3,P4,P5,P6,P7,P8;
  
      P(N,0.0,0.0,1.0);
      P(P1,-2.0,-2.0,1.0);
      P(P2,-2.0,-2.0,-1.0);
      P(S,0.0,0.0,-1.0);
      P(P3,2.0,-2.0,1.0);
      P(P4,2.0,-2.0,-1.0);
      P(P5,2.0,2.0,1.0);
      P(P6,2.0,2.0,-1.0);
      P(P7,-2.0,2.0,1.0);
      P(P8,-2.0,2.0,-1.0);
        
      triple p0[]={N,P1,P2,S,
                   N,P3,P4,S,
                   N,P5,P6,S,
                   N,P7,P8,S,
                   N,P1,P2,S,
                   N,P3,P4,S};
   
      out->addSurface(2,3,3,4,p0,uknot,vknot,m,Weights);
      out->addSurface(2,3,3,4,p0+4,uknot,vknot,m,Weights);
      if(!half) {
        out->addSurface(2,3,3,4,p0+8,uknot,vknot,m,Weights);
        out->addSurface(2,3,3,4,p0+12,uknot,vknot,m,Weights);
      }
      
      break;
    }
    default:
      reportError("Invalid sphere type");
  }
  
  return true;
}

bool drawCylinder::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  
  out->addCylinder(1.0,1.0,m,NULL,NULL,NULL,1.0,T);
  
  return true;
}
  
bool drawDisk::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  
  out->addDisk(1.0,m,NULL,NULL,NULL,1.0,T);
  
  return true;
}
  
bool drawTube::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  PRCmaterial m(ambient,diffuse,emissive,specular,opacity,shininess);
  
  Int n=center.length();
  
  if(center.piecewisestraight()) {
    triple *centerControls=new(UseGC) triple[n+1];
    for(Int i=0; i <= n; ++i)
      centerControls[i]=center.point(i);
    size_t N=n+1;
    triple *controls=new(UseGC) triple[N];
    for(Int i=0; i <= n; ++i)
      controls[i]=g.point(i);
    out->addTube(N,centerControls,controls,true,m);
  } else {
    size_t N=3*n+1;
    triple *centerControls=new(UseGC) triple[N];
    centerControls[0]=center.point((Int) 0);
    centerControls[1]=center.postcontrol((Int) 0);
    size_t k=1;
    for(Int i=1; i < n; ++i) {
      centerControls[++k]=center.precontrol(i);
      centerControls[++k]=center.point(i);
      centerControls[++k]=center.postcontrol(i);
    }
    centerControls[++k]=center.precontrol(n);
    centerControls[++k]=center.point(n);
    
    triple *controls=new(UseGC) triple[N];
    controls[0]=g.point((Int) 0);
    controls[1]=g.postcontrol((Int) 0);
    k=1;
    for(Int i=1; i < n; ++i) {
      controls[++k]=g.precontrol(i);
      controls[++k]=g.point(i);
      controls[++k]=g.postcontrol(i);
    }
    controls[++k]=g.precontrol(n);
    controls[++k]=g.point(n);
    
    out->addTube(N,centerControls,controls,false,m);
  }
      
  return true;
}

bool drawPixel::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  out->addPoint(v,c,width);
  
  return true;
}
  
void drawPixel::render(GLUnurbs *nurb, double size2,
                       const triple& Min, const triple& Max,
                       double perspective, bool lighton, bool transparent) 
{
#ifdef HAVE_GL
  if(invisible)
    return;
  
  static GLfloat V[4];

  glEnable(GL_COLOR_MATERIAL);
  glColorMaterial(GL_FRONT_AND_BACK,GL_EMISSION);
  
  static GLfloat Black[]={0,0,0,1};
  glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,Black);
  glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,Black);
  glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,Black);
  glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);

  glPointSize(1.0+width);
  
  glBegin(GL_POINT);
  storecolor(V,0,c);
  glColor4fv(V);
  store(V,v);
  glVertex3fv(V);
  glEnd();
  
  glPointSize(1.0);
  glDisable(GL_COLOR_MATERIAL);
#endif
}

const string drawBaseTriangles::wrongsize=
  "triangle indices require 3 components";
const string drawBaseTriangles::outofrange="index out of range";

void drawBaseTriangles::bounds(const double* t, bbox3& b)
{
  double x,y,z;
  double X,Y,Z;
  triple* tP;

  if(t == NULL) tP=P;
  else {
    tP=new triple[nP];
    for(size_t i=0; i < nP; i++)
      tP[i]=t*P[i];
  }

  boundstriples(x,y,z,X,Y,Z,nP,tP);

  b.add(x,y,z);
  b.add(X,Y,Z);

  if(t == NULL) {
    Min=triple(x,y,z);
    Max=triple(X,Y,Z);
  } else delete[] tP;
}

void drawBaseTriangles::ratio(const double* t, pair &b,
                              double (*m)(double, double), double fuzz,
                              bool &first)
{
  triple* tP;

  if(t == NULL) tP=P;
  else {
    tP=new triple[nP];
    for(size_t i=0; i < nP; i++)
      tP[i]=t*P[i];
  }

  ratiotriples(b,m,first,nP,tP);
  
  if(t != NULL)
    delete[] tP;
}

bool drawTriangles::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;
  
  if (nC) {
    const RGBAColour white(1,1,1,opacity);
    const RGBAColour black(0,0,0,opacity);
    const PRCmaterial m(black,white,black,specular,opacity,PRCshininess);
    out->addTriangles(nP,P,nI,PI,m,nN,N,NI,0,NULL,NULL,nC,C,CI,0,NULL,NULL,30);
  } else {
    const PRCmaterial m(ambient,diffuse,emissive,specular,opacity,PRCshininess);
    out->addTriangles(nP,P,nI,PI,m,nN,N,NI,0,NULL,NULL,0,NULL,NULL,0,NULL,NULL,30);
  }

  return true;
}

void drawTriangles::render(GLUnurbs *nurb, double size2, const triple& Min,
                           const triple& Max, double perspective, bool lighton,
                           bool transparent)
{
#ifdef HAVE_GL
  if(invisible)
    return;

  if(invisible || ((diffuse.A < 1.0) ^ transparent)) return;

  triple m,M;
  static double t[16]; // current transform
  glGetDoublev(GL_MODELVIEW_MATRIX,t);
  run::transpose(t,4);

  bbox3 B(this->Min,this->Max);
  B.transform(t);

  m=B.Min();
  M=B.Max();

  if(perspective) {
    const double f=m.getz()*perspective;
    const double F=M.getz()*perspective;
    if((M.getx() < min(f*Min.getx(),F*Min.getx()) ||
        m.getx() > max(f*Max.getx(),F*Max.getx()) ||
        M.gety() < min(f*Min.gety(),F*Min.gety()) ||
        m.gety() > max(f*Max.gety(),F*Max.gety()) ||
        M.getz() < Min.getz() ||
        m.getz() > Max.getz()))
      return;
  } else {
    if((M.getx() < Min.getx() || m.getx() > Max.getx() ||
        M.gety() < Min.gety() || m.gety() > Max.gety() ||
        M.getz() < Min.getz() || m.getz() > Max.getz()))
      return;
  }

  setcolors(nC,!nC,diffuse,ambient,emissive,specular,shininess);
  if(!nN) lighton=false;
  
  glBegin(GL_TRIANGLES);
  for(size_t i=0; i < nI; i++) {
    const uint32_t *pi=PI[i];
    const uint32_t *ni=NI[i];
    const uint32_t *ci=nC ? CI[i] : 0;
    if(lighton)
      glNormal3f(N[ni[0]].getx(),N[ni[0]].gety(),N[ni[0]].getz());
    if(nC)
      glColor4f(C[ci[0]].R,C[ci[0]].G,C[ci[0]].B,C[ci[0]].A);
    glVertex3f(P[pi[0]].getx(),P[pi[0]].gety(),P[pi[0]].getz());
    if(lighton)
      glNormal3f(N[ni[1]].getx(),N[ni[1]].gety(),N[ni[1]].getz());
    if(nC)
      glColor4f(C[ci[1]].R,C[ci[1]].G,C[ci[1]].B,C[ci[1]].A);
    glVertex3f(P[pi[1]].getx(),P[pi[1]].gety(),P[pi[1]].getz());
    if(lighton)
      glNormal3f(N[ni[2]].getx(),N[ni[2]].gety(),N[ni[2]].getz());
    if(nC)
      glColor4f(C[ci[2]].R,C[ci[2]].G,C[ci[2]].B,C[ci[2]].A);
    glVertex3f(P[pi[2]].getx(),P[pi[2]].gety(),P[pi[2]].getz());
  }
  glEnd();

  if(nC)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

} //namespace camp

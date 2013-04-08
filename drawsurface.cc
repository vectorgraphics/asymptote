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
  }
  
  if(colors) {
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
    Triple *Vertices;
    if(t == NULL) Vertices=vertices;
    else {
      static Triple buf[4];
      Vertices=buf;
      transformTriples(t,4,Vertices,vertices);
    }
  
    double *v=Vertices[0];
    x=v[0];
    y=v[1];
    z=v[2];
    X=x;
    Y=y;
    Z=z;
    for(size_t i=1; i < 4; ++i) {
      double *v=Vertices[i];
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
    Triple* Controls;
    if(t == NULL) Controls=controls;
    else {
      static Triple buf[16];
      Controls=buf;
      transformTriples(t,16,Controls,controls);
    }

    static double c1[16];

    for(int i=0; i < 16; ++i)
      c1[i]=Controls[i][0];
    double c0=c1[0];
    double fuzz=sqrtFuzz*run::norm(c1,16);
    x=bound(c1,min,b.empty ? c0 : min(c0,b.left),fuzz);
    X=bound(c1,max,b.empty ? c0 : max(c0,b.right),fuzz);
    
    for(int i=0; i < 16; ++i)
      c1[i]=Controls[i][1];
    c0=c1[0];
    fuzz=sqrtFuzz*run::norm(c1,16);
    y=bound(c1,min,b.empty ? c0 : min(c0,b.bottom),fuzz);
    Y=bound(c1,max,b.empty ? c0 : max(c0,b.top),fuzz);
    
    for(int i=0; i < 16; ++i)
      c1[i]=Controls[i][2];
    c0=c1[0];
    fuzz=sqrtFuzz*run::norm(c1,16);
    z=bound(c1,min,b.empty ? c0 : min(c0,b.lower),fuzz);
    Z=bound(c1,max,b.empty ? c0 : max(c0,b.upper),fuzz);
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
  if(straight) {
    Triple *Vertices;
    if(t == NULL) Vertices=vertices;
    else {
      static Triple buf[4];
      Vertices=buf;
      transformTriples(t,4,Vertices,vertices);
    }
  
    if(first) {
      first=false;
      double *ci=Vertices[0];
      triple v=triple(ci[0],ci[1],ci[2]);
      b=pair(xratio(v),yratio(v));
    }
  
    double x=b.getx();
    double y=b.gety();
    for(size_t i=0; i < 4; ++i) {
      double *ci=Vertices[i];
      triple v=triple(ci[0],ci[1],ci[2]);
      x=m(x,xratio(v));
      y=m(y,yratio(v));
    }
    b=pair(x,y);
  } else {
    Triple* Controls;
    if(t == NULL) Controls=controls;
    else {
      static Triple buf[16];
      Controls=buf;
      transformTriples(t,16,Controls,controls);
    }

    static triple c3[16];
    for(int i=0; i < 16; ++i) {
      double *ci=Controls[i];
      c3[i]=triple(ci[0],ci[1],ci[2]);
    }
  
    if(first) {
      triple v=c3[0];
      b=pair(xratio(v),yratio(v));
      first=false;
    }
  
    b=pair(bound(c3,m,xratio,b.getx(),fuzz),bound(c3,m,yratio,b.gety(),fuzz));
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

inline triple displacement1(const Triple& z0, const Triple& c0,
                           const Triple& c1, const Triple& z1)
{
  triple Z0(z0);
  triple Z1(z1);
  return maxabs(displacement(triple(c0[0],c0[1],c0[2]),Z0,Z1),
                displacement(triple(c1[0],c1[1],c1[2]),Z0,Z1));
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
    
    bbox3 B(this->Min,this->Max);
    B.transform(t);
  
    m=B.Min();
    M=B.Max();
  }

  if(perspective) {
    const double f=m.getz()*perspective;
    const double F=M.getz()*perspective;
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
    
  setcolors(colors,lighton,diffuse,ambient,emissive,specular,shininess);
  
  const triple size3(s*(Max.getx()-Min.getx()),s*(Max.gety()-Min.gety()),
                     Max.getz()-Min.getz());
  
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
  const size_t n=nu*nv;
  Triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    Controls=new Triple[n];
    transformTriples(t,n,Controls,controls);
  }

  double *v=Controls[0];
  double x=v[0];
  double y=v[1];
  double z=v[2];
  double X=x;
  double Y=y;
  double Z=z;
  for(size_t i=1; i < n; ++i) {
    double *v=Controls[i];
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
  Triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    Controls=new Triple[n];
    transformTriples(t,n,Controls,controls);
  }

  if(first) {
    first=false;
    double *ci=Controls[0];
    triple v=triple(ci[0],ci[1],ci[2]);
    b=pair(xratio(v),yratio(v));
  }
  
  double x=b.getx();
  double y=b.gety();
  for(size_t i=0; i < n; ++i) {
    double *ci=Controls[i];
    triple v=triple(ci[0],ci[1],ci[2]);
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

void drawSphere::P(Triple& t, double x, double y, double z)
{
  if(half) {
    double temp=z; z=x; x=-temp;
  }
  
  double f=T[3]*x+T[7]*y+T[11]*z+T[15];
  if(f == 0.0) run::dividebyzero();
  f=1.0/f;
  
  t[0]=(T[0]*x+T[4]*y+T[ 8]*z+T[12])*f;
  t[1]=(T[1]*x+T[5]*y+T[ 9]*z+T[13])*f;
  t[2]=(T[2]*x+T[6]*y+T[10]*z+T[14])*f;
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
  
      Triple N,S,P1,P2,P3,P4,P5,P6,P7,P8;
  
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
        
      Triple p0[]=
        {{N[0],N[1],N[2]},
         {P1[0],P1[1],P1[2]},
         {P2[0],P2[1],P2[2]},
         {S[0],S[1],S[2]},
     
         {N[0],N[1],N[2]},
         {P3[0],P3[1],P3[2]},
         {P4[0],P4[1],P4[2]},
         {S[0],S[1],S[2]},
     
         {N[0],N[1],N[2]},
         {P5[0],P5[1],P5[2]},
         {P6[0],P6[1],P6[2]},
         {S[0],S[1],S[2]},
  
         {N[0],N[1],N[2]},
         {P7[0],P7[1],P7[2]},
         {P8[0],P8[1],P8[2]},
         {S[0],S[1],S[2]},
     
         {N[0],N[1],N[2]},
         {P1[0],P1[1],P1[2]},
         {P2[0],P2[1],P2[2]},
         {S[0],S[1],S[2]},
     
         {N[0],N[1],N[2]},
         {P3[0],P3[1],P3[2]},
         {P4[0],P4[1],P4[2]},
         {S[0],S[1],S[2]},
        };

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
    Triple *centerControls=new(UseGC) Triple[n+1];
    for(Int i=0; i <= n; ++i)
      store(centerControls[i],center.point(i));
    size_t N=n+1;
    Triple *controls=new(UseGC) Triple[N];
    for(Int i=0; i <= n; ++i)
      store(controls[i],g.point(i));
    out->addTube(N,centerControls,controls,true,m,NULL,NULL,NULL,1.0);
  } else {
    size_t N=3*n+1;
    Triple *centerControls=new(UseGC) Triple[N];
    store(centerControls[0],center.point((Int) 0));
    store(centerControls[1],center.postcontrol((Int) 0));
    size_t k=1;
    for(Int i=1; i < n; ++i) {
      store(centerControls[++k],center.precontrol(i));
      store(centerControls[++k],center.point(i));
      store(centerControls[++k],center.postcontrol(i));
    }
    store(centerControls[++k],center.precontrol(n));
    store(centerControls[++k],center.point(n));
    
    Triple *controls=new(UseGC) Triple[N];
    store(controls[0],g.point((Int) 0));
    store(controls[1],g.postcontrol((Int) 0));
    k=1;
    for(Int i=1; i < n; ++i) {
      store(controls[++k],g.precontrol(i));
      store(controls[++k],g.point(i));
      store(controls[++k],g.postcontrol(i));
    }
    store(controls[++k],g.precontrol(n));
    store(controls[++k],g.point(n));
    
    out->addTube(N,centerControls,controls,false,m,NULL,NULL,NULL,1.0);
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
  Triple* tP;

  if(t == NULL) tP=P;
  else {
    tP=new Triple[nP];
    transformTriples(t,nP,tP,P);
  }

  boundsTriples(x,y,z,X,Y,Z,nP,tP);

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
  Triple* tP;

  if(t == NULL) tP=P;
  else {
    tP=new Triple[nP];
    transformTriples(t,nP,tP,P);
  }

  ratioTriples(b,m,first,nP,tP);
  
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
      glNormal3f(N[ni[0]][0],N[ni[0]][1],N[ni[0]][2]);
    if(nC)
      glColor4f(C[ci[0]].R,C[ci[0]].G,C[ci[0]].B,C[ci[0]].A);
    glVertex3f(P[pi[0]][0],P[pi[0]][1],P[pi[0]][2]);
    if(lighton)
      glNormal3f(N[ni[1]][0],N[ni[1]][1],N[ni[1]][2]);
    if(nC)
      glColor4f(C[ci[1]].R,C[ci[1]].G,C[ci[1]].B,C[ci[1]].A);
    glVertex3f(P[pi[1]][0],P[pi[1]][1],P[pi[1]][2]);
    if(lighton)
      glNormal3f(N[ni[2]][0],N[ni[2]][1],N[ni[2]][2]);
    if(nC)
      glColor4f(C[ci[2]].R,C[ci[2]].G,C[ci[2]].B,C[ci[2]].A);
    glVertex3f(P[pi[2]][0],P[pi[2]][1],P[pi[2]][2]);
  }
  glEnd();

  if(nC)
    glDisable(GL_COLOR_MATERIAL);
#endif
}

} //namespace camp

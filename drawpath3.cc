/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"

namespace camp {

using vm::array;
  
bool drawPath3::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  Int n=g.length();
  if(n == 0 || invisible)
    return true;

  if(straight) {
    triple *controls=new(UseGC) triple[n+1];
    for(Int i=0; i <= n; ++i)
      controls[i]=g.point(i);
    
    out->addLine(n+1,controls,color);
  } else {
    int m=3*n+1;
    triple *controls=new(UseGC) triple[m];
    controls[0]=g.point((Int) 0);
    controls[1]=g.postcontrol((Int) 0);
    size_t k=1;
    for(Int i=1; i < n; ++i) {
      controls[++k]=g.precontrol(i);
      controls[++k]=g.point(i);
      controls[++k]=g.postcontrol(i);
    }
    controls[++k]=g.precontrol(n);
    controls[++k]=g.point(n);
    out->addBezierCurve(m,controls,color);
  }
  
  return true;
}

void drawPath3::render(GLUnurbs *nurb, double, const triple&, const triple&,
                       double, bool lighton, bool transparent)
{
#ifdef HAVE_GL
  Int n=g.length();
  if(n == 0 || invisible || ((color.A < 1.0) ^ transparent))
    return;

  bool havebillboard=interaction == BILLBOARD;
  
  GLfloat Diffuse[]={0.0,0.0,0.0,(GLfloat) color.A};
  glMaterialfv(GL_FRONT,GL_DIFFUSE,Diffuse);
  
  static GLfloat Black[]={0.0,0.0,0.0,1.0};
  glMaterialfv(GL_FRONT,GL_AMBIENT,Black);
    
  GLfloat Emissive[]={(GLfloat) color.R,(GLfloat) color.G,(GLfloat) color.B,
		      (GLfloat) color.A};
  glMaterialfv(GL_FRONT,GL_EMISSION,Emissive);
    
  glMaterialfv(GL_FRONT,GL_SPECULAR,Black);
  
  glMaterialf(GL_FRONT,GL_SHININESS,128.0);
  
  if(havebillboard) BB.init();
  
  if(straight) {
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      if(havebillboard) {
        static GLfloat controlpoints[3];
        BB.store(controlpoints,v,center);
        glVertex3fv(controlpoints);
      } else
        glVertex3f(v.getx(),v.gety(),v.getz());
    }
    glEnd();
  } else {
    for(Int i=0; i < n; ++i) {
      static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
      static GLfloat controlpoints[12];
      if(havebillboard) {
        BB.store(controlpoints,g.point(i),center);
        BB.store(controlpoints+3,g.postcontrol(i),center);
        BB.store(controlpoints+6,g.precontrol(i+1),center);
        BB.store(controlpoints+9,g.point(i+1),center);
      } else {
        store(controlpoints,g.point(i));
        store(controlpoints+3,g.postcontrol(i));
        store(controlpoints+6,g.precontrol(i+1));
        store(controlpoints+9,g.point(i+1));
      }
      
      gluBeginCurve(nurb);
      gluNurbsCurve(nurb,8,knots,3,controlpoints,4,GL_MAP1_VERTEX_3);
      gluEndCurve(nurb);
    }
  }
#endif
}

drawElement *drawPath3::transformed(const double* t)
{
  return new drawPath3(t,this);
}
  
bool drawNurbsPath3::write(prcfile *out, unsigned int *, double, groupsmap&)
{
  if(invisible)
    return true;

  out->addCurve(degree,n,controls,knots,color,weights);
  
  return true;
}

// Approximate bounds by bounding box of control polyhedron.
void drawNurbsPath3::bounds(const double* t, bbox3& b)
{
  double x,y,z;
  double X,Y,Z;
  
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

drawElement *drawNurbsPath3::transformed(const double* t)
{
  return new drawNurbsPath3(t,this);
}

void drawNurbsPath3::ratio(const double* t, pair &b, double (*m)(double, double),
                           double, bool &first)
{
  triple* Controls;
  if(t == NULL) Controls=controls;
  else {
    Controls=new triple[n];
    for(size_t i=0; i < n; i++)
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

void drawNurbsPath3::displacement()
{
#ifdef HAVE_GL
  size_t nknots=degree+n+1;
  if(Controls == NULL) {
    Controls=new(UseGC)  GLfloat[(weights ? 4 : 3)*n];
    Knots=new(UseGC) GLfloat[nknots];
  }
  if(weights)
    for(size_t i=0; i < n; ++i)
      store(Controls+4*i,controls[i],weights[i]);
  else
    for(size_t i=0; i < n; ++i)
      store(Controls+3*i,controls[i]);
  
  for(size_t i=0; i < nknots; ++i)
    Knots[i]=knots[i];
#endif  
}

void drawNurbsPath3::render(GLUnurbs *nurb, double, const triple&,
                            const triple&, double, bool lighton,
                            bool transparent)
{
#ifdef HAVE_GL
  if(invisible || ((color.A < 1.0) ^ transparent))
    return;
  
  GLfloat Diffuse[]={0.0,0.0,0.0,(GLfloat) color.A};
  glMaterialfv(GL_FRONT,GL_DIFFUSE,Diffuse);
  
  static GLfloat Black[]={0.0,0.0,0.0,1.0};
  glMaterialfv(GL_FRONT,GL_AMBIENT,Black);
    
  GLfloat Emissive[]={(GLfloat) color.R,(GLfloat) color.G,(GLfloat) color.B,
		      (GLfloat) color.A};
  glMaterialfv(GL_FRONT,GL_EMISSION,Emissive);
    
  glMaterialfv(GL_FRONT,GL_SPECULAR,Black);
  
  glMaterialf(GL_FRONT,GL_SHININESS,128.0);
  
  if(weights)
    gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex4fv);
  else gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex3fv);

  gluBeginCurve(nurb);
  int order=degree+1;
  gluNurbsCurve(nurb,order+n,Knots,weights ? 4 : 3,Controls,order,
                weights ? GL_MAP1_VERTEX_4 : GL_MAP1_VERTEX_3);
  gluEndCurve(nurb);
  
  if(weights)
    gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex3fv);
#endif
}

} //namespace camp

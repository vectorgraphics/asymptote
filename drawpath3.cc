/*****
 * drawpath3.cc
 *
 * Stores a path3 that has been added to a picture.
 *****/

#include "drawpath3.h"

namespace camp {

using vm::array;
  
bool drawPath3::write(prcfile *out, unsigned int *count, array *, array *)
{
  Int n=g.length();
  if(n == 0 || invisible)
    return true;

  ostringstream buf;
  
  if(straight) {
    if(name == "")
      buf << "line-" << count[LINE]++;
    else
      buf << name;
    
    controls=new(UseGC) Triple[n+1];
    for(Int i=0; i <= n; ++i)
      store(controls[i],g.point(i));
    out->add(new PRCline(out,n+1,controls,color,scale3D,name.c_str()));
  } else {
    if(name == "")
      buf << "curve-" << count[CURVE]++;
    else
      buf << name;
    
    int m=3*n+1;
    controls=new(UseGC) Triple[m];
    store(controls[0],g.point((Int) 0));
    store(controls[1],g.postcontrol((Int) 0));
    size_t k=1;
    for(Int i=1; i < n; ++i) {
      store(controls[++k],g.precontrol(i));
      store(controls[++k],g.point(i));
      store(controls[++k],g.postcontrol(i));
    }
    store(controls[++k],g.precontrol((Int) n));
    store(controls[++k],g.point((Int) n));
    out->add(new PRCBezierCurve(out,3,m,controls,color,name));
  }
  return true;
}

void drawPath3::render(GLUnurbs *nurb, double, const triple&, const triple&,
                       double, bool transparent)
{
#ifdef HAVE_LIBGL
  Int n=g.length();
  if(n == 0 || invisible || ((color.A < 1.0) ^ transparent))
    return;

  GLfloat Diffuse[]={0.0,0.0,0.0,color.A};
  glMaterialfv(GL_FRONT,GL_DIFFUSE,Diffuse);
  
  static GLfloat Black[]={0.0,0.0,0.0,1.0};
  glMaterialfv(GL_FRONT,GL_AMBIENT,Black);
    
  GLfloat Emissive[]={color.R,color.G,color.B,color.A};
  glMaterialfv(GL_FRONT,GL_EMISSION,Emissive);
    
  glMaterialfv(GL_FRONT,GL_SPECULAR,Black);
  
  glMaterialf(GL_FRONT,GL_SHININESS,128.0);
  
  if(straight) {
    glBegin(GL_LINE_STRIP);
    for(Int i=0; i <= n; ++i) {
      triple v=g.point(i);
      glVertex3f(v.getx(),v.gety(),v.getz());
    }
    glEnd();
  } else {
    for(Int i=0; i < n; ++i) {
      static GLfloat knots[8]={0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0};
      static GLfloat controlpoints[12];
      store(controlpoints,g.point(i));
      store(controlpoints+3,g.postcontrol(i));
      store(controlpoints+6,g.precontrol(i+1));
      store(controlpoints+9,g.point(i+1));
    
      gluBeginCurve(nurb);
      gluNurbsCurve(nurb,8,knots,3,controlpoints,4,GL_MAP1_VERTEX_3);
      gluEndCurve(nurb);
    }
  }
#endif
}

drawElement *drawPath3::transformed(const array& t)
{
  return new drawPath3(t,this);
}
  
bool drawNurbsPath3::write(prcfile *out, unsigned int *count, array *index,
                           array *origin)
{
  ostringstream buf;
  if(name == "") 
    buf << "curve-" << count[CURVE]++;
  else
    buf << name;
  
  if(invisible)
    return true;

  out->add(new PRCcurve(out,udegree,nu,controls,uknots,color,scale3D,
                        weights != NULL,weights,name.c_str()));
  
  return true;
}

// Approximate bounds by bounding box of control polyhedron.
void drawNurbsPath3::bounds(bbox3& b)
{
  double *v=controls[0];
  double x=v[0];
  double y=v[1];
  double z=v[2];
  double X=x;
  double Y=y;
  double Z=z;
  for(size_t i=1; i < nu; ++i) {
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

drawElement *drawNurbsPath3::transformed(const array& t)
{
  return new drawNurbsPath3(t,this);
}

void drawNurbsPath3::ratio(pair &b, double (*m)(double, double), bool &first)
{
  if(first) {
    first=false;
    double *ci=controls[0];
    triple v=triple(ci[0],ci[1],ci[2]);
    b=pair(xratio(v),yratio(v));
  }
  
  double x=b.getx();
  double y=b.gety();
  for(size_t i=0; i < nu; ++i) {
    double *ci=controls[i];
    triple v=triple(ci[0],ci[1],ci[2]);
    x=m(x,xratio(v));
    y=m(y,yratio(v));
  }
  b=pair(x,y);
}

void drawNurbsPath3::displacement()
{
#ifdef HAVE_LIBGL
  if(weights)
    for(size_t i=0; i < nu; ++i)
      store(Controls+4*i,controls[i],weights[i]);
  else
    for(size_t i=0; i < nu; ++i)
      store(Controls+3*i,controls[i]);
  
  size_t nuknotsm1=udegree+nu;
  for(size_t i=0; i <= nuknotsm1; ++i)
    uKnots[i]=uknots[i];
#endif  
}

void drawNurbsPath3::render(GLUnurbs *nurb, double, const triple&,
                            const triple&, double, bool transparent)
{
#ifdef HAVE_LIBGL
  if(invisible || ((color.A < 1.0) ^ transparent))
    return;
  
  GLfloat Diffuse[]={0.0,0.0,0.0,color.A};
  glMaterialfv(GL_FRONT,GL_DIFFUSE,Diffuse);
  
  static GLfloat Black[]={0.0,0.0,0.0,1.0};
  glMaterialfv(GL_FRONT,GL_AMBIENT,Black);
    
  GLfloat Emissive[]={color.R,color.G,color.B,color.A};
  glMaterialfv(GL_FRONT,GL_EMISSION,Emissive);
    
  glMaterialfv(GL_FRONT,GL_SPECULAR,Black);
  
  glMaterialf(GL_FRONT,GL_SHININESS,128.0);
  
  if(weights)
    gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex4fv);
  else gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex3fv);

  gluBeginCurve(nurb);
  int uorder=udegree+1;
  gluNurbsCurve(nurb,uorder+nu,uKnots,weights ? 4 : 3,Controls,uorder,
                weights ? GL_MAP1_VERTEX_4 : GL_MAP1_VERTEX_3);
  gluEndCurve(nurb);
  
  if(weights)
    gluNurbsCallback(nurb,GLU_NURBS_VERTEX,(_GLUfuncptr) glVertex3fv);
#endif
}

} //namespace camp

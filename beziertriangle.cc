/*****
 * drawbeziertriangle.cc
 *
 * Stores a Bezier triangle that has been added to a picture.
 *****/

#include "drawsurface.h"

namespace camp {

const double pixel=1.0; // Adaptive rendering constant.

GLuint nvertices;
unsigned int nindices;
const unsigned int NBUFFER=10000000; // FIXME
GLfloat buffer[NBUFFER]; // Move into class.
GLuint indices[NBUFFER];

// Store the normal vector given the directional derivatives bu in the u
// direction and bv in the v direction.
inline void normal(const triple& bu, const triple& bv, GLuint index)
{
  triple n=unit(triple(bu.gety()*bv.getz()-bu.getz()*bv.gety(),
                       bu.getz()*bv.getx()-bu.getx()*bv.getz(),
                       bu.getx()*bv.gety()-bu.gety()*bv.getx()));
  GLfloat *p=buffer+6*index;
  *p++=n.getx();
  *p++=n.gety();
  *p=n.getz();
}

double size2;
triple size3; // Move to class.

// Store the vertex.
GLuint vertex(triple V)
{
  GLfloat *p=buffer+6*nvertices+3;
  *p++=V.getx();
  *p++=V.gety();
  *p=V.getz();
  return nvertices++;
}

void mesh(const triple *p, const GLuint *I)
{
  //bool lighton=true; // TODO
  // Draw the frame of the control points of a cubic Bezier mesh

  GLuint I0=I[0];
  GLuint I1=I[1];
  GLuint I2=I[2];

  normal(-p[0]+p[1],-p[0]+p[2],I0);
  normal(-p[3]+p[6],-p[3]+p[7],I1);
  normal(-p[5]+p[8],-p[5]+p[9],I2);

  GLuint *q=indices+nindices;
  *q++=I0;
  *q++=I1;
  *q=I2;

  nindices += 3; // Can this be made more reliable?
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

triple displacement(const triple *controls)
{
  triple d=drawElement::zero;

  triple z0=controls[0];
  triple z1=controls[6];
  triple z2=controls[9];

  // Optimize straight & planar cases.

  for(size_t i=1; i < 10; ++i)
    d=maxabs(d,displacement2(controls[i],z0,unit(cross(z1-z0,z2-z0))));
  d=maxabs(d,10*displacement2(controls[4],z0,unit(cross(z1-z0,z2-z0))));

  d=maxabs(d,4*displacement1(controls[0],controls[1],controls[3],controls[6]));
  d=maxabs(d,4*displacement1(controls[0],controls[2],controls[5],controls[9]));
  d=maxabs(d,4*displacement1(controls[6],controls[7],controls[8],controls[9]));

  // TODO: calculate displacement d from interior
  return d;
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


void render(const triple *p, int n,
            GLuint I0, GLuint I1, GLuint I2, // Ii is the index to Pi
            triple P0, triple P1, triple P2, // Pi is the (double) version of Ii
            bool flat1, bool flat2, bool flat3, // Flatness flags for each boundary.
            string pos) {
  // Uses a uniform partition
  // p points to an array of 10 triples.
  // Draw a Bezier triangle.
  // p is the set of control points for the Bezier triangle
  // n is the maximum number of iterations to compute
  triple d=displacement(p);

  // This is the previous method, but it involves fewer triangle computations at
  // the end (since if the surface is sufficiently flat, it just draws the
  // sufficiently flat triangle, rather than trying to properly utilize the
  // already computed values.
  //
  // Ideally, this increase in redundancy will me mitigated by a smarter render
  // using the tree-like structure (still being developed).

  if(n == 0 || fraction(d,size3)*size2 < pixel) { // If triangle is flat...
    GLuint pp[]={I0,I1,I2};

    /*
     *cout << pos << " (" << I0 << ", " << I1 << ", " << I2 << ")" << endl <<
     *  P0 << ", " << P1 << ", " << P2 << "-" << endl << "(" <<
     *  buffer[6*I0+3] << "," << buffer[6*I0+4] << "," << buffer[6*I0+5]+20 << "), (" <<
     *  buffer[6*I1+3] << "," << buffer[6*I1+4] << "," << buffer[6*I1+5]+20 << "), (" <<
     *  buffer[6*I2+3] << "," << buffer[6*I2+4] << "," << buffer[6*I2+5]+20 << ")" << endl;
     */

    /*
     *cout << "write(\"" << pos  << "(" << I0 << ", " << I1 << ", " << I2 << ")\");" << endl <<
     *  "write(" << P0 << "-(" << buffer[6*I0+3] << "," << buffer[6*I0+4] << "," << buffer[6*I0+5]+20 << "));" <<
     *  "write(" << P1 << "-(" << buffer[6*I1+3] << "," << buffer[6*I1+4] << "," << buffer[6*I1+5]+20 << "));" <<
     *  "write(" << P2 << "-(" << buffer[6*I2+3] << "," << buffer[6*I2+4] << "," << buffer[6*I2+5]+20 << "));" << endl;
     */
    mesh(p,pp);
  } else { // Triangle is not flat

    /*    Naming Convention:
     *
     *                            030
     *                           /\
     *                          /  \
     *                         /    \
     *                        /      \
     *                       /   up   \
     *                      /          \
     *                     /            \
     *                    /              \
     *               pp2 /________________\ pp3
     *                  /\               / \
     *                 /  \             /   \
     *                /    \           /     \
     *               /      \   mid   /       \
     *              /        \       /         \
     *             /          \     /           \
     *            /    left    \   /    right    \
     *           /              \ /               \
     *          /________________V_________________\
     *       003                 pp1                 300
     */

    // Subdivide triangle
    triple l003=p[0];
    triple p102=p[1];
    triple p012=p[2];
    triple p201=p[3];
    triple p111=p[4];
    triple p021=p[5];
    triple r300=p[6];
    triple p210=p[7];
    triple p120=p[8];
    triple u030=p[9];

    triple u021=0.5*(u030+p021);
    triple u120=0.5*(u030+p120);

    triple p033=0.5*(p021+p012);
    triple p231=0.5*(p120+p111);
    triple p330=0.5*(p120+p210);

    triple p123=0.5*(p012+p111);

    triple l012=0.5*(p012+l003);
    triple p312=0.5*(p111+p201);
    triple r210=0.5*(p210+r300);

    triple l102=0.5*(l003+p102);
    triple p303=0.5*(p102+p201);
    triple r201=0.5*(p201+r300);

    triple u012=0.5*(u021+p033);
    triple u210=0.5*(u120+p330);
    triple l021=0.5*(p033+l012);
    triple p4xx=0.5*p231+0.25*(p111+p102);
    triple r120=0.5*(p330+r210);
    triple px4x=0.5*p123+0.25*(p111+p210);
    triple pxx4=0.25*(p021+p111)+0.5*p312;
    triple l201=0.5*(l102+p303);
    triple r102=0.5*(p303+r201);

    triple l210=0.5*(px4x+l201); // = m120
    triple r012=0.5*(px4x+r102); // = m021
    triple l300=0.5*(l201+r102); // = r003 = m030

    triple r021=0.5*(pxx4+r120); // = m012
    triple u201=0.5*(u210+pxx4); // = m102
    triple r030=0.5*(u210+r120); // = u300 = m003

    triple u102=0.5*(u012+p4xx); // = m201
    triple l120=0.5*(l021+p4xx); // = m210
    triple l030=0.5*(u012+l021); // = u003 = m300

    triple l111=0.5*(p123+l102);
    triple r111=0.5*(p312+r210);
    triple u111=0.5*(u021+p231);
    triple m111=0.25*(p033+p330+p303+p111);

    //  For each edge of the triangle
    //    - Check for flatness
    //    - Store points in the GLU array accordingly
    GLuint a1, a2, a3;
    triple pp1 = 0.5*(P1+P0);
    triple pp2 = 0.5*(P2+P0);
    triple pp3 = 0.5*(P2+P1);

    // A kludge to remove subdivision cracks (if it is indeed rounding error)
    //double epsilon=1e-4; // This is to be made adaptive to the zoom-level

    // How epsilon was computed: guess-and-check...
    const double epsilon=0.1;///1e-2;
    double epsilon1=epsilon/(fraction(pp1-u030,size3)*size2);
    double epsilon2=epsilon/(fraction(pp2-r300,size3)*size2);
    double epsilon3=epsilon/(fraction(pp3-l003,size3)*size2);
    //cout << epsilon1/epsilon << "," << epsilon2/epsilon << "," << epsilon3/epsilon <<
       //endl << pp1-u030 << "," << pp2-r300 << "," << pp3-l003 << endl << "-----" << endl;

    // Add the epsilon adjustments to the computed vertices.
    pp1 += epsilon1*(pp1-u030);
    pp2 += epsilon2*(pp2-r300);
    pp3 += epsilon3*(pp3-l003);

    // This method works reasonably well, although it's still possible to see
    // mesh overlaps at high zoom levels.
    //double epsilon=1e-4; // This is to be made adaptive to the zoom-level
    //pp1 += epsilon*(pp1-u030);
    //pp2 += epsilon*(pp2-r300);
    //pp3 += epsilon*(pp3-l003);

     /*
      *cout << [>pos << endl <<<]
      *  "write(" << pp1 << "- (" <<
      *  0.5*((GLfloat) P1.getx() + (GLfloat) P0.getx()) << "," <<
      *  0.5*((GLfloat) P1.gety() + (GLfloat) P0.gety()) << "," <<
      *  0.5*((GLfloat) P1.getz() + (GLfloat) P0.getz()) << "));" << endl <<
      *  "write(" << pp2 << "- (" <<
      *  0.5*((GLfloat) P2.getx() + (GLfloat) P0.getx()) << "," <<
      *  0.5*((GLfloat) P2.gety() + (GLfloat) P0.gety()) << "," <<
      *  0.5*((GLfloat) P2.getz() + (GLfloat) P0.getz()) << "));" << endl <<
      *  "write(" << pp3 << "- (" <<
      *  0.5*((GLfloat) P2.getx() + (GLfloat) P1.getx()) << "," <<
      *  0.5*((GLfloat) P2.gety() + (GLfloat) P1.gety()) << "," <<
      *  0.5*((GLfloat) P2.getz() + (GLfloat) P1.getz()) << "));" << endl;
      */

    if(flat1 || fraction(displacement1(p[0],p[1],p[3],p[6]),size3)*size2 < pixel/4) {
      flat1=true;
      a1=vertex(pp1);
    } else {
      a1=vertex(l300);
    }
    if(flat2 || fraction(displacement1(p[0],p[2],p[5],p[9]),size3)*size2 < pixel/4) {
      flat2=true;
      a2=vertex(pp2);
    } else {
      a2=vertex(l030);
    }
    if(flat3 || fraction(displacement1(p[6],p[7],p[8],p[9]),size3)*size2 < pixel/4) {
      flat3 = true;
      a3=vertex(pp3);
    } else {
      a3=vertex(r030);
    }

    triple l[]={l003,l102,l012,l201,l111,l021,l300,l210,l120,l030}; // left
    triple r[]={l300,r102,r012,r201,r111,r021,r300,r210,r120,r030}; // right
    triple u[]={l030,u102,u012,u201,u111,u021,r030,u210,u120,u030}; // up
    triple m[]={r030,u201,r021,u102,m111,r012,l030,l120,l210,l300}; // middle

    --n;
    render(l,n,I0,a1,a2,
           P0,
           flat1 ? pp1 : l300,
           flat2 ? pp2 : l030,
           flat1,flat2,false,
           pos +"l");
    render(r,n,a1,I1,a3,
           flat1 ? pp1 : l300,
           P1,
           flat3 ? pp3 : r030,
           flat1,false,flat3,
           pos +"r");
    render(u,n,a2,a3,I2,
           flat2 ? pp2 : l030,
           flat3 ? pp3 : r030,
           P2,
           false,flat2,flat3,
           pos +"u");
    render(m,n,a3,a2,a1,
           flat3 ? pp3 : r030,
           flat2 ? pp2 : l030,
           flat1 ? pp1 : l300,
           false,false,false,
           pos +"m");
  }
}

void render(const triple *p, int n) {
  if(n > 0) {
    //cout << "-------------BEGIN-----------------" << endl;
    render(p,n,vertex(p[0]),vertex(p[6]),vertex(p[9]),p[0],p[6],p[9],false,false,false,"");
    //cout << "############  end  ################" << endl;
  } else {
    GLuint I[]={vertex(p[0]),vertex(p[6]),vertex(p[9])};
    mesh(p,I);
  }
}

/*void renderBisec(const triple *p, int n, GLuint I0, GLuint I1, GLuint I2) {
  // Uses a bisection method
  // Draw a semi-accurate rendition of a Bezier triangle.
  // p is the set of control points for the Bezier triangle
  // n is the number of iterations to compute
  triple b0_003=p[0];
  triple b0_102=p[1];
  triple b0_012=p[2];
  triple b0_201=p[3];
  triple b0_111=p[4];
  triple b0_021=p[5];
  triple b0_300=p[6];
  triple b0_210=p[7];
  triple b0_120=p[8];
  triple b0_030=p[9];

  triple b1_002=0.5*(b0_003+b0_012);
  triple b1_101=0.5*(b0_111+b0_102);
  triple b1_011=0.5*(b0_021+b0_012);
  triple b1_200=0.5*(b0_210+b0_201);
  triple b1_110=0.5*(b0_120+b0_111);
  triple b1_020=0.5*(b0_021+b0_030);

  triple b2_001=0.5*(b1_011+b1_002);
  triple b2_100=0.5*(b1_110+b1_101);
  triple b2_010=0.5*(b1_020+b1_011);

  triple b3_000=0.5*(b2_001+b2_010);


  //trip p0_003=b0_030
  //trip p0_102=b1_020
  //trip p0_012=b0_120
  //trip p0_201=b2_010
  //trip p0_111=b1_110
  //trip p0_021=b0_210
  //trip p0_300=b3_000
  //trip p0_210=b2_100
  //trip p0_120=b1_200
  //trip p0_030=b0_300

  triple p1_002=0.5*(b0_030+b0_120);
  triple p1_101=0.5*(b1_110+b1_020);
  triple p1_011=0.5*(b0_210+b0_120);
  triple p1_200=0.5*(b2_100+b2_010);
  triple p1_110=0.5*(b1_200+b1_110);
  triple p1_020=0.5*(b0_210+b0_300);

  triple p2_001=0.5*(p1_011+p1_002);
  triple p2_100=0.5*(p1_110+p1_101);
  triple p2_010=0.5*(p1_020+p1_011);

  triple p3_000=0.5*(p2_001+p2_010);


  //trip q0_003=b0_300
  //trip q0_102=b1_200
  //trip q0_012=b0_201
  //trip q0_201=b2_100
  //trip q0_111=b1_101
  //trip q0_021=b0_102
  //trip q0_300=b3_000
  //trip q0_210=b2_001
  //trip q0_120=b1_002
  //trip q0_030=b0_003

  triple q1_002=0.5*(b0_300+b0_201);
  triple q1_101=0.5*(b1_101+b1_200);
  triple q1_011=0.5*(b0_102+b0_201);
  triple q1_200=0.5*(b2_001+b2_100);
  triple q1_110=0.5*(b1_002+b1_101);
  triple q1_020=0.5*(b0_102+b0_003);

  triple q2_001=0.5*(q1_011+q1_002);
  triple q2_100=0.5*(q1_110+q1_101);
  triple q2_010=0.5*(q1_020+q1_011);

  triple q3_000=0.5*(q2_001+q2_010);

  GLuint a6=vertex(q3_000);
  GLuint a9=vertex(b3_000);
  GLuint c6=vertex(p3_000);

  triple t1[]={b0_003,q1_020,b1_002,q2_010,q1_110,b2_001,q3_000,q2_100,
               q1_200,b3_000};
  triple t2[]={b3_000,q1_200,b2_100,q2_100,q1_101,b1_200,q3_000,q2_001,
               q1_002,b0_300};
  triple t3[]={b0_300,p1_020,b1_200,p2_010,p1_110,b2_100,p3_000,p2_100,
               p1_200,b3_000};
  triple t4[]={b3_000,p1_200,b2_010,p2_100,p1_101,b1_020,p3_000,p2_001,
               p1_002,b0_030};
  if(n > 1) {
    --n;
    renderBisec(t1,n,I0,a6,a9);
    renderBisec(t2,n,a9,a6,I1);
    renderBisec(t3,n,I1,c6,a9);
    renderBisec(t4,n,a9,c6,I2);
  } else {
    GLuint a7=vertex(q2_100);
    GLuint a8=vertex(q1_200);
    //GLuint l1[]={I0,vertex(q1_020),vertex(b1_002),vertex(q2_010),vertex(q1_110),
                 //vertex(b2_001),a6,a7,a8,a9};
    GLuint l1[]={I0,a6,a9};
    mesh(t1,l1);

    GLuint b2=vertex(b2_100);
    GLuint b5=vertex(b1_200);
    //GLuint l2[]={a9,a8,b2,a7,vertex(q1_101),b5,
                 //a6,vertex(q2_001),vertex(q1_002),I1};
    GLuint l2[]={a9,a6,I1};
    mesh(t2,l2);

    GLuint c7=vertex(p2_100);
    GLuint c8=vertex(p1_200);
    //GLuint l3[]={I1,vertex(p1_020),b5,vertex(p2_010),vertex(p1_110),b2,c6,c7,c8,a9};
    GLuint l3[]={I1,b2,a9};
    mesh(t3,l3);

    //GLuint l4[]={a9,c8,vertex(b2_010),c7,vertex(p1_101),vertex(b1_020),c6,vertex(p2_001), vertex(p1_002),I2};
    GLuint l4[]={a9,c6,I2};
    mesh(t4,l4);
  }
}

void renderBisec(const triple *p, int n=2) {
  if(n > 0)
    renderBisec(p,n,vertex(p[0]),vertex(p[6]),vertex(p[9]));
  else {
    GLuint I[]={vertex(p[0]),vertex(p[1]),vertex(p[2]),vertex(p[3]),vertex(p[4]),
                vertex(p[5]),vertex(p[6]),vertex(p[7]),vertex(p[8]),vertex(p[9])};
    mesh(p,I);
  }
}*/

void bezierTriangle(const triple *g, double Size2, triple Size3)
{
  size2=Size2;
  size3=Size3;
  //cout << "size2=" << size2 << endl << "size3=" << size3 << endl;
  //size2=1100.69796038695;
  //size3=triple(10.605274364509,10.021984274461,18.1865334794732);
  //bool lighton=true;
  //size_t nI=1;

  /*triple g[]={
    triple(0,2,0),
    triple(1,0,0),triple(1,1,0),
    triple(2,0,0),triple(2,1,0),triple(2,2,0),
    triple(3,0,0),triple(3,1,0),triple(3,2,0),triple(3,3,0)
  };*/

  /*triple g[]={
    // This is an example of the flatness test failing; to be precise, the test should
    // not be explicitly for flatness. (Set p_111 to a nonzero z-value to get another
    // issue).
    triple(0,0,0),
    triple(-1,0,0),triple(1,1,0),
    triple(4,0,0),triple(-2,1,0),triple(2,2,0),
    triple(3,0,0),triple(3,-1,0),triple(3,4,0),triple(3,3,0)
  };*/

  /*triple g[]={
     // Here the tessellation can easily be seen through the reflection of light.
     // Perhaps what is needed is a 'curvature test'?
    triple(0,0,0),
    triple(1,0,0),triple(1,1,0),
    triple(2,0,0),triple(2,1,2),triple(2,2,-1),
    triple(3,0,0),triple(3,1,0),triple(3,2,-1),triple(3,3,0)
  };*/

  /*triple g[]={
    triple(0,0,0),
    triple(1,0,0),triple(1,1,0),
    triple(2,0,0),triple(2,1,2),triple(2,2,0),
    triple(3,0,0),triple(3,1,0),triple(3,2,0),triple(3,3,0)
  };*/

  /*
  triple g[]={
    triple(0,0,0),
    triple(1,0,0),triple(0.5,sqrt(3)/2,0),
    triple(2,0,0),triple(1.5,sqrt(3)/2,2),triple(1,sqrt(3),0),
    triple(3,0,0),triple(2.5,sqrt(3)/2,0),triple(2,sqrt(3),0),triple(1.5,3*sqrt(3)/2,0)
  };
  */

  /*triple g[]={
    triple(0,1,0),
    triple(1,0,0),triple(1,1,0),
    triple(2,0,0),triple(2,1,1),triple(2,2,0),
    triple(3,0,0),triple(3,1,1),triple(3,2,1),triple(3,3,0)
  };*/

  /*triple g[]={
    triple(0,0,0),
    triple(1,0,0),triple(1,1,0),
    triple(2,0,0),triple(2,1,0),triple(2,2,0),
    triple(3,0,0),triple(3,3,1),triple(3,0,1),triple(3,3,0)
  };*/

  int n=8; // Number of iterations (doubled for single-bisection)

  //for(int i=0; i < 10000; ++i) {
  nvertices=0;
  nindices=0;

  //  for(int j=0; j < 10; ++j)
  //    g[j] += triple(i*0.0001,0,0);
  render(g,n); // uniform
  //renderBisec(g,n); // bisection
  /*for(int i = 0; i < nvertices; ++i)
    cout << buffer[i] << endl;*/
  /*for(GLuint i = 0; 6*i < nvertices; ++i)
    cout << i << "::::" << endl <<
      buffer[6*i] << "," << buffer[6*i+1] << "," << buffer[6*i+2] << endl <<
      buffer[6*i+3] << "," << buffer[6*i+4] << "," << buffer[6*i+5]+20 << endl << endl;
  cout << "..." << endl;*/
  /*for(int i = 0; 3*i < nindices; i+=3)
    cout << "i:" << i << endl << indices[i] << "," << indices[i+1] << "," << indices[i+2] << endl;*/
  //  for(int j=0; j < 10; ++j)
  //    g[j] += triple(1,1,1);
  //renderBisec(g,n); // double bisection

  size_t size=6*sizeof(GL_FLOAT);

  glEnableClientState(GL_NORMAL_ARRAY);
  glEnableClientState(GL_VERTEX_ARRAY);
  glNormalPointer(GL_FLOAT,size,buffer);
  glVertexPointer(3,GL_FLOAT,size,buffer+3);
  glDrawElements(GL_TRIANGLES,nindices,GL_UNSIGNED_INT,indices);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
//  }
  //cout << "triangles=" << nindices/3 << endl;
  //cout << "nvertices (buffer)=" << nvertices << endl;
  //unsigned int side=3*(1 << n)+1;
  //cout << "nvertices (unique)=" << side*(side+1)/2 << endl;
}

} //namespace camp

// Gilles Dumoulin's C++ port of Paul Bourke's triangulation code available
// from http://astronomy.swin.edu.au/~pbourke/papers/triangulate
// Used with permission of Paul Bourke.
// Segmentation fault and numerical precision fixes by John C. Bowman

#include "Delaunay.h"

inline double max(double a, double b)
{
  return (a > b) ? a : b;
}

////////////////////////////////////////////////////////////////////////
// CircumCircle():
//   Return true if a point (xp,yp) is inside the circumcircle made up
//   of the points (x1,y1), (x2,y2), (x3,y3)
//   The circumcircle centre is returned in (xc,yc) and the radius r
//   Note: A point on the edge is inside the circumcircle
////////////////////////////////////////////////////////////////////////

Int CircumCircle(double xp, double yp, double x1, double y1, double x2, 
		 double y2, double x3, double y3, double &xc, double &yc,
		 double &r)
{
  double m1, m2, mx1, mx2, my1, my2;
  double dx, dy, rsqr, drsqr;

  double eps=100.0*DBL_EPSILON*max(max(fabs(y1),fabs(y2)),fabs(y3));
  
/* Check for coincident points */
  if(fabs(y1 - y2) <= eps && fabs(y2 - y3) <= eps)
    return(false);
  if(fabs(y2 - y1) <= eps) {
    m2 = - (x3 - x2) / (y3 - y2);
    mx2 = (x2 + x3) / 2.0;
    my2 = (y2 + y3) / 2.0;
    xc = (x2 + x1) / 2.0;
    yc = m2 * (xc - mx2) + my2;
  } else if(fabs(y3 - y2) <= eps) {
    m1 = - (x2 - x1) / (y2 - y1);
    mx1 = (x1 + x2) / 2.0;
    my1 = (y1 + y2) / 2.0;
    xc = (x3 + x2) / 2.0;
    yc = m1 * (xc - mx1) + my1;
  } else {
    m1 = - (x2 - x1) / (y2 - y1); 
    m2 = - (x3 - x2) / (y3 - y2); 
    mx1 = (x1 + x2) / 2.0; 
    mx2 = (x2 + x3) / 2.0;
    my1 = (y1 + y2) / 2.0;
    my2 = (y2 + y3) / 2.0;
    xc = (m1 * mx1 - m2 * mx2 + my2 - my1) / (m1 - m2); 
    yc = m1 * (xc - mx1) + my1; 
  }
  dx = x2 - xc;
  dy = y2 - yc;
  rsqr = dx * dx + dy * dy;
  r = sqrt(rsqr); 
  dx = xp - xc;
  dy = yp - yc;
  drsqr = dx * dx + dy * dy;
  return drsqr <= rsqr;
}

int XYZCompare(const void *v1, const void *v2) 
{
  XYZ *p1, *p2;
    
  p1 = (XYZ*)v1;
  p2 = (XYZ*)v2;
  if(p1->x < p2->x)
    return(-1);
  else if(p1->x > p2->x)
    return(1);
  else
    return(0);
}

///////////////////////////////////////////////////////////////////////////////
// Triangulate():
//   Triangulation subroutine
//   Takes as input NV vertices in array pxyz
//   Returned is a list of ntri triangular faces in the array v
//   These triangles are arranged in a consistent clockwise order.
//   The triangle array v should be allocated to 3 * nv
//   The vertex array pxyz must be big enough to hold 3 additional points.
//   By default, the array pxyz is automatically presorted and postsorted.
///////////////////////////////////////////////////////////////////////////////

Int Triangulate(Int nv, XYZ pxyz[], ITRIANGLE v[], Int &ntri,
		bool presort, bool postsort)
{
  Int *complete = NULL;
  IEDGE *edges = NULL; 
  IEDGE *p_EdgeTemp;
  Int nedge = 0;
  Int trimax, emax = 200;
  Int inside;
  Int i, j, k;
  double xp, yp, x1, y1, x2, y2, x3, y3, xc, yc, r;
  double xmin, xmax, ymin, ymax, xmid, ymid;
  double dx, dy, dmax; 

  if(presort) qsort(pxyz,nv,sizeof(XYZ),XYZCompare);
  else postsort=false;
  
/* Allocate memory for the completeness list, flag for each triangle */
  trimax = 4 * nv;
  complete = new Int[trimax];
/* Allocate memory for the edge list */
  edges = new IEDGE[emax];
/*
  Find the maximum and minimum vertex bounds.
  This is to allow calculation of the bounding triangle
*/
  xmin = pxyz[0].x;
  ymin = pxyz[0].y;
  xmax = xmin;
  ymax = ymin;
  for(i = 1; i < nv; i++) {
    XYZ *pxyzi=pxyz+i;
    if (pxyzi->x < xmin) xmin = pxyz[i].x;
    if (pxyzi->x > xmax) xmax = pxyz[i].x;
    if (pxyzi->y < ymin) ymin = pxyz[i].y;
    if (pxyzi->y > ymax) ymax = pxyz[i].y;
  }
  dx = xmax - xmin;
  dy = ymax - ymin;
  dmax = (dx > dy) ? dx : dy;
  xmid = (xmax + xmin) / 2.0;
  ymid = (ymax + ymin) / 2.0;
/*
  Set up the supertriangle
  his is a triangle which encompasses all the sample points.
  The supertriangle coordinates are added to the end of the
  vertex list. The supertriangle is the first triangle in
  the triangle list.
*/
  pxyz[nv+0].x = xmid - 20 * dmax;
  pxyz[nv+0].y = ymid - dmax;
  pxyz[nv+1].x = xmid;
  pxyz[nv+1].y = ymid + 20 * dmax;
  pxyz[nv+2].x = xmid + 20 * dmax;
  pxyz[nv+2].y = ymid - dmax;
  v->p1 = nv;
  v->p2 = nv+1;
  v->p3 = nv+2;
  complete[0] = false;
  ntri = 1;
/*
  Include each point one at a time into the existing mesh
*/
  for(i = 0; i < nv; i++) {
    xp = pxyz[i].x;
    yp = pxyz[i].y;
    nedge = 0;
/*
  Set up the edge buffer.
  If the point (xp,yp) lies inside the circumcircle then the
  three edges of that triangle are added to the edge buffer
  and that triangle is removed.
*/
    for(j = 0; j < ntri; j++) {
      if(complete[j])
	continue;
      ITRIANGLE *vj=v+j;
      x1 = pxyz[vj->p1].x;
      y1 = pxyz[vj->p1].y;
      x2 = pxyz[vj->p2].x;
      y2 = pxyz[vj->p2].y;
      x3 = pxyz[vj->p3].x;
      y3 = pxyz[vj->p3].y;
      inside = CircumCircle(xp, yp, x1, y1, x2, y2, x3, y3, xc, yc, r);
      if (xc + r < xp)
// Suggested
// if (xc + r + eps < xp)
	complete[j] = true;
      if(inside) {
/* Check that we haven't exceeded the edge list size */
	if(nedge + 3 >= emax) {
	  emax += 100;
	  p_EdgeTemp = new IEDGE[emax];
	  for (Int i = 0; i < nedge; i++) {
	    p_EdgeTemp[i] = edges[i];   
	  }
	  delete []edges;
	  edges = p_EdgeTemp;
	}
	ITRIANGLE *vj=v+j;
	edges[nedge+0].p1 = vj->p1;
	edges[nedge+0].p2 = vj->p2;
	edges[nedge+1].p1 = vj->p2;
	edges[nedge+1].p2 = vj->p3;
	edges[nedge+2].p1 = vj->p3;
	edges[nedge+2].p2 = vj->p1;
	nedge += 3;
	v[j] = v[ntri-1];
	complete[j] = complete[ntri-1];
	ntri--;
	j--;
      }
    }
/*
  Tag multiple edges
  Note: if all triangles are specified anticlockwise then all
  interior edges are opposite pointing in direction.
*/
    for(j = 0; j < nedge - 1; j++) {
      for(k = j + 1; k < nedge; k++) {
	if((edges[j].p1 == edges[k].p2) && (edges[j].p2 == edges[k].p1)) {
	  edges[j].p1 = -1;
	  edges[j].p2 = -1;
	  edges[k].p1 = -1;
	  edges[k].p2 = -1;
	}
	/* Shouldn't need the following, see note above */
	if((edges[j].p1 == edges[k].p1) && (edges[j].p2 == edges[k].p2)) {
	  edges[j].p1 = -1;
	  edges[j].p2 = -1;
	  edges[k].p1 = -1;
	  edges[k].p2 = -1;
	}
      }
    }
/*
  Form new triangles for the current point
  Skipping over any tagged edges.
  All edges are arranged in clockwise order.
*/
    for(j = 0; j < nedge; j++) {
      if(edges[j].p1 < 0 || edges[j].p2 < 0)
	continue;
      v[ntri].p1 = edges[j].p1;
      v[ntri].p2 = edges[j].p2;
      v[ntri].p3 = i;
      complete[ntri] = false;
      ntri++;
    }
  }
/*
  Remove triangles with supertriangle vertices
  These are triangles which have a vertex number greater than nv
*/
  for(i = 0; i < ntri; i++) {
    ITRIANGLE *vi=v+i;
    if(vi->p1 >= nv || vi->p2 >= nv || vi->p3 >= nv) {
      *vi = v[ntri-1];
      ntri--;
      i--;
    }
  }
  delete[] edges;
  delete[] complete;

  if(postsort) { 
    for(i = 0; i < ntri; i++) {
      ITRIANGLE *vi=v+i;
      vi->p1=pxyz[vi->p1].i;
      vi->p2=pxyz[vi->p2].i;
      vi->p3=pxyz[vi->p3].i;
    }
  }

  return 0;
} 

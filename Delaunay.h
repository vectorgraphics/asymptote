#ifndef Delaunay_H
#define Delaunay_H

#include <iostream>
#include <stdlib.h> // for C qsort 
#include <cmath>
#include <cfloat>

#include "common.h"

struct ITRIANGLE{
  Int p1, p2, p3;
};

struct IEDGE{
  Int p1, p2;
};

struct XYZ{
  double x, y;
	Int i;
};

Int Triangulate(Int nv, XYZ pxyz[], ITRIANGLE v[], Int &ntri,
		bool presort=true, bool postsort=true);
Int CircumCircle(double, double, double, double, double, double, double, 
double, double&, double&, double&);

#endif



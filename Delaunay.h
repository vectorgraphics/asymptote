#ifndef Delaunay_H
#define Delaunay_H

#include <iostream>
#include <stdlib.h> // for C qsort 
#include <cmath>
#include <cfloat>

struct ITRIANGLE{
  int p1, p2, p3;
};

struct IEDGE{
  int p1, p2;
};

struct XYZ{
  double x, y;
	int i;
};

int XYZCompare(const void *v1, const void *v2);
int Triangulate(int nv, XYZ pxyz[], ITRIANGLE v[], int &ntri,
		bool presort=true, bool postsort=true);
int CircumCircle(double, double, double, double, double, double, double, 
double, double&, double&, double&);

#endif



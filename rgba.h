#pragma once

#ifdef HAVE_LIBTIRPC
#include "xstream.h"
#endif

namespace prc {

struct RGBAColour
{
  RGBAColour(double r=0.0, double g=0.0, double b=0.0, double a=1.0) :
    R(r), G(g), B(b), A(a) {}
  double R,G,B,A;

  void Set(double r, double g, double b, double a=1.0)
  {
    R = r; G = g; B = b; A = a;
  }
  bool operator==(const RGBAColour &c) const
  {
    return (R==c.R && G==c.G && B==c.B && A==c.A);
  }
  bool operator!=(const RGBAColour &c) const
  {
    return !(R==c.R && G==c.G && B==c.B && A==c.A);
  }
  bool operator<(const RGBAColour &c) const
  {
    if(R!=c.R)
      return (R<c.R);
    if(G!=c.G)
      return (G<c.G);
    if(B!=c.B)
      return (B<c.B);
    return (A<c.A);
  }

  friend RGBAColour operator * (const RGBAColour& a, const double d)
  { return RGBAColour(a.R*d,a.G*d,a.B*d,a.A*d); }
  friend RGBAColour operator * (const double d, const RGBAColour& a)
  { return RGBAColour(a.R*d,a.G*d,a.B*d,a.A*d); }

#ifdef HAVE_LIBTIRPC
  friend xdr::oxstream& operator<<(xdr::oxstream& out, RGBAColour const& col)
  {
    out << (float) col.R << (float) col.G << (float) col.B << (float) col.A;
    return out;
  }
#endif
};

}

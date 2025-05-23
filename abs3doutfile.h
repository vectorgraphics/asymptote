#ifndef ABS3DOUTFILE_H
#define ABS3DOUTFILE_H

#include "common.h"
#include "triple.h"
#include "prcfile.h"
#include "material.h"

namespace camp
{

inline bool distinct(const uint32_t *I, const uint32_t *J)
{
  return I[0] != J[0] || I[1] != J[1] || I[2] != J[2];
}

class abs3Doutfile : public gc {
protected:
  bool singleprecision;
  string KEY;
public:
  abs3Doutfile(bool singleprecision=false) : singleprecision(singleprecision),
                                             KEY("") {}
  virtual ~abs3Doutfile()=default;

  void setKEY(const string& KEY) {this->KEY=KEY;}

  virtual void close()=0;

  virtual void addPatch(triple const* controls, prc::RGBAColour const* c)=0;

  virtual void addStraightPatch(
          triple const* controls, prc::RGBAColour const* c)=0;

  virtual void addBezierTriangle(
          triple const* controls, prc::RGBAColour const* c)=0;

  virtual void addStraightBezierTriangle(
          triple const* controls, prc::RGBAColour const* c)=0;

#ifdef HAVE_LIBGLM
  virtual void addMaterial(Material const& mat)=0;
#endif

  virtual void addSphere(triple const& center, double radius)=0;

  virtual void addHemisphere(triple const& center, double radius, double const& polar, double const& azimuth)=0;

  virtual void addCylinder(triple const& center, double radius, double height,
                           double const& polar, const double& azimuth,
                           bool core)=0;

  virtual void addDisk(triple const& center, double radius,
                       double const& polar, const double& azimuth)=0;

  virtual void addTube(const triple* g, double width, bool core)=0;

  virtual void addTriangles(size_t nP, const triple* P, size_t nN,
                            const triple* N, size_t nC, const prc::RGBAColour* C,
                            size_t nI, const uint32_t (* PI)[3],
                            const uint32_t (* NI)[3],
                            const uint32_t (* CI)[3])=0;

  virtual void addCurve(const triple& z0, const triple& c0,
                        const triple& c1, const triple& z1)=0;

  virtual void addCurve(const triple& z0, const triple& z1)=0;

  virtual void addPixel(const triple& z0, double width)=0;

  virtual void precision(int digits)=0;

};

}

#endif

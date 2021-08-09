/*
 * v3dfile.h
 * Header file for v3d export and types
 * Written by: Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com> \
 *   and John C. Bowman <bowman@ualberta.ca>
 */

#ifndef V3DFILE_H
#define V3DFILE_H

#include <prc/oPRCFile.h>

#include "common.h"
#include "abs3doutfile.h"
#include "xstream.h"
#include "triple.h"
#include "material.h"
#include "glrender.h"
#include "enumheaders/v3dtypes.h"
#include "enumheaders/v3dheadertypes.h"

namespace camp
{

class AHeader
{
protected:
  virtual uint32_t getByteSize() const = 0;
  virtual void writeContent(xdr::oxstream& ox) const = 0;

public:
  explicit AHeader(v3dheadertypes const& ty) : ty(ty) {}
  virtual ~AHeader() = default;
  friend xdr::oxstream& operator<< (xdr::oxstream& ox, AHeader const& header);
private:
  v3dheadertypes ty;
};

template<typename T, uint32_t n=sizeof(T)>
class SingleObjectHeader : public AHeader
{
public:
  explicit SingleObjectHeader(v3dheadertypes const& ty, T const& ob) : AHeader(ty), obj(ob)
  {
  }
  ~SingleObjectHeader() override = default;

protected:
  uint32_t getByteSize() const override
  {
    return max((uint32_t)1, (uint32_t)(n / 4));
  }

  void writeContent(xdr::oxstream &ox) const override
  {
    ox << obj;
  }

private:
  T obj;
};

using open_mode=xdr::xios::open_mode;
using TripleHeader=SingleObjectHeader<triple, 3*8>;
using PairHeader=SingleObjectHeader<pair, 2*8>;
using DoubleFloatHeader=SingleObjectHeader<double>;
using Uint32Header=SingleObjectHeader<uint32_t>;

const unsigned int v3dVersion=0;

enum v3dTriangleIndexType : uint32_t
{
  index_Pos=0,
  index_PosNorm=1,
  index_PosColor=2,
  index_PosNormColor=3,
};

class absv3dfile : public abs3Doutfile
{
public:
  absv3dfile();

  void writeInit();
  void finalize();

  void addPatch(triple const* controls, triple const& Min, triple const& Max, prc::RGBAColour const* c) override;
  void addStraightPatch(
          triple const* controls, triple const& Min, triple const& Max, prc::RGBAColour const* c) override;
  void addBezierTriangle(
          triple const* control, triple const& Min, triple const& Max, prc::RGBAColour const* c) override;
  void addStraightBezierTriangle(
          triple const* controls, triple const& Min, triple const& Max, prc::RGBAColour const* c) override;


  void addMaterial(Material const& mat) override;

  void addSphere(triple const& center, double radius) override;
  void addSphereHalf(triple const& center, double radius, double const& polar, double const& azimuth) override;

  void addCylinder(triple const& center, double radius, double height,
                   double const& polar, const double& azimuth,
                   bool core) override;
  void addDisk(triple const& center, double radius,
               double const& polar, const double& azimuth) override;
  void addTube(const triple *g, double width,
               const triple& Min, const triple& Max, bool core) override;

  void addTriangles(size_t nP, const triple* P, size_t nN,
                            const triple* N, size_t nC, const prc::RGBAColour* C,
                            size_t nI, const uint32_t (*PI)[3],
                            const uint32_t (*NI)[3], const uint32_t (*CI)[3],
                            const triple& Min, const triple& Max) override;

  void addCurve(triple const& z0, triple const& c0, triple const& c1, triple const& z1, triple const& Min,
                triple const& Max) override;

  void addCurve(triple const& z0, triple const& z1, triple const& Min, triple const& Max) override;

  void addPixel(triple const& z0, double width, triple const& Min, triple const& Max) override;

  void precision(int digits) override;


protected:
  void addvec4(glm::vec4 const& vec);
  void addCenterIndexMat();
  void addIndices(uint32_t const* trip);
  void addTriples(triple const* triples, size_t n);
  void addColors(prc::RGBAColour const* col, size_t nc);

  void addHeaders();
  void addCenters();

  virtual xdr::oxstream& getXDRFile() = 0;

private:
  bool finalized;
};

class v3dfile : public absv3dfile
{
public:
  explicit v3dfile(string const& name, open_mode mode=xdr::xios::open_mode::out);
  ~v3dfile() override;

protected:
  void close() override;
  void closeFile();

  xdr::oxstream& getXDRFile() override;

private:
  xdr::oxstream xdrfile;
  bool finished;
};

class memv3dfile : public absv3dfile
{
public:
  memv3dfile();
  ~memv3dfile() override;

  [[nodiscard]]
  char const* data() const;

protected:
  void close() override;
  xdr::oxstream& getXDRFile() override;

private:
  xdr::memoxstream memxdrfile;
  bool finalized;
};

} //namespace camp
#endif

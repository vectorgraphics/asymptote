/************
*
*   This file is part of a tool for producing 3D content in the PRC format.
*   Copyright (C) 2008  Orest Shardt <shardtor (at) gmail dot com>
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU Lesser General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public License
*   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*************/

#ifndef __O_PRC_FILE_H
#define __O_PRC_FILE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "PRC.h"
#include "PRCbitStream.h"
#include "writePRC.h"

class oPRCFile;

enum Entity {LINE=0,CURVE,SURFACE,nEntity};

std::string Name(oPRCFile *p, Entity e, std::string s);

struct RGBAColour
{
  RGBAColour(double r=0.0, double g=0.0, double b=0.0, double a=1.0) :
    R(r), G(g), B(b), A(a) {}
  double R,G,B,A;

  bool operator==(const RGBAColour &c) const
  {
    return (R==c.R && G==c.G && B==c.B && A==c.A);
  }
};

struct PRCMaterial
{
  PRCMaterial() : alpha(1.0),shininess(1.0) {}
  PRCMaterial(const RGBAColour &a, const RGBAColour &d, const RGBAColour &e,
              const RGBAColour &s, double p, double h) :
      ambient(a), diffuse(d), emissive(e), specular(s), alpha(p), shininess(h) {}
  RGBAColour ambient,diffuse,emissive,specular;
  double alpha,shininess;

  bool operator==(const PRCMaterial &m) const
  {
    return (ambient==m.ambient && diffuse==m.diffuse && emissive==m.emissive
        && specular==m.specular && shininess==m.shininess);
  }
};

class PRCentity
{
  public:
    PRCentity(oPRCFile *p,const RGBAColour &c, double scale = 1.0,
              std::string name = "") : 
      colour(c), useMaterial(false), scale(scale), name(name), parent(p) {}
    PRCentity(oPRCFile *p,const PRCMaterial &m, double scale = 1.0,
              std::string name = "") : 
      material(m), useMaterial(true), scale(scale), name(name), parent(p) {}
    uint32_t getGraphicsIndex();
    virtual void writeRepresentationItem(PRCbitStream&,uint32_t) = 0;
    virtual void writeTopologicalContext(PRCbitStream&) = 0;
    virtual void writeExtraGeometryContext(PRCbitStream&) = 0;
    RGBAColour colour;
    PRCMaterial material;
    bool useMaterial;
    double scale;
    std::string name;
  virtual ~PRCentity() {}
  protected:
    oPRCFile *parent;
};

class PRCsurface : public PRCentity
{
  public:
    // contructor with colour
    PRCsurface(oPRCFile *p, uint32_t dU, uint32_t dV, uint32_t nU, uint32_t nV,
	       double cP[][3], double *kU, double *kV, const RGBAColour &c,
	       double scale = 1.0, bool iR = false, double w[] = 0,
	       double g = 0, std::string name = "") :
      PRCentity(p,c,scale,Name(p,SURFACE,name)), degreeU(dU), degreeV(dV),
      numberOfControlPointsU(nU), numberOfControlPointsV(nV),
      knotsU(kU), knotsV(kV), controlPoints(cP),
      isRational(iR), weights(w), granularity(g) {}
    // contructor with material
    PRCsurface(oPRCFile *p, uint32_t dU, uint32_t dV, uint32_t nU, uint32_t nV,
	       double cP[][3], double *kU, double *kV, const PRCMaterial &m,
	       double scale = 1.0, bool iR = false, double w[] = 0,
	       double g = 0, std::string name = "") :
      PRCentity(p,m,scale,Name(p,SURFACE,name)), degreeU(dU), degreeV(dV),
      numberOfControlPointsU(nU), numberOfControlPointsV(nV),
      knotsU(kU), knotsV(kV), controlPoints(cP),
      isRational(iR), weights(w), granularity(g) {}
    virtual void writeRepresentationItem(PRCbitStream&,uint32_t);
    virtual void writeTopologicalContext(PRCbitStream&);
    virtual void writeExtraGeometryContext(PRCbitStream&);
  protected:
    virtual void writeKnots(PRCbitStream &out)
    {
      for(uint32_t i = 0; i < degreeU+numberOfControlPointsU+1; ++i)
        out << knotsU[i];
      for(uint32_t i = 0; i < degreeV+numberOfControlPointsV+1; ++i)
        out << knotsV[i];
    }
  private:
    uint32_t degreeU, degreeV, numberOfControlPointsU, numberOfControlPointsV;
    double *knotsU, *knotsV;
    double (*controlPoints)[3];
    bool isRational;
    double *weights;
    double granularity;
};

class PRCline : public PRCentity
{
  public:
    // constructor with colour
    PRCline(oPRCFile *p, uint32_t n, double P[][3], const RGBAColour &c,
            double scale = 1.0, std::string name="") :
      PRCentity(p,c,scale,Name(p,LINE,name)), numberOfPoints(n), points(P) {}
    // constructor with material
    PRCline(oPRCFile *p, uint32_t n, double P[][3], const PRCMaterial &m,
            double scale = 1.0, std::string name = "") :
      PRCentity(p,m,scale,Name(p,LINE,name)), numberOfPoints(n), points(P) {}
      virtual void writeRepresentationItem(PRCbitStream&,uint32_t);
      virtual void writeTopologicalContext(PRCbitStream&);
      virtual void writeExtraGeometryContext(PRCbitStream&);
  private:
    uint32_t numberOfPoints;
    double (*points)[3];
};

class PRCcurve : public PRCentity
{
  public:
    // constructor with colour
    PRCcurve(oPRCFile *p, uint32_t d, uint32_t n, double cP[][3], double *k,
	     const RGBAColour &c, double scale = 1.0, bool iR = false,
	     double w[] = 0, std::string name="") :
      PRCentity(p,c,scale,Name(p,CURVE,name)), degree(d),
      numberOfControlPoints(n), knots(k), controlPoints(cP), isRational(iR),
      weights(w) {}
    // constructor with material
    PRCcurve(oPRCFile *p, uint32_t d, uint32_t n, double cP[][3], double *k,
	     const PRCMaterial &m, double scale = 1.0, bool iR = false,
	     double w[] = 0, std::string name = "") :
      PRCentity(p,m,scale,Name(p,CURVE,name)), degree(d),
      numberOfControlPoints(n), knots(k), controlPoints(cP), isRational(iR),
      weights(w) {}
    virtual void writeRepresentationItem(PRCbitStream&,uint32_t);
    virtual void writeTopologicalContext(PRCbitStream&);
    virtual void writeExtraGeometryContext(PRCbitStream&);
  protected:
    virtual void writeKnots(PRCbitStream &out)
    {
      for(uint32_t i = 0; i < degree+numberOfControlPoints+1; ++i)
        out << knots[i];
    }
  private:
    uint32_t degree, numberOfControlPoints;
    double *knots;
    double (*controlPoints)[3];
    bool isRational;
    double *weights;
};

class PRCCompressedSection
{
  public:
    PRCCompressedSection(oPRCFile *p) : data(0),prepared(false),parent(p),
        out(data,0) {}
    virtual ~PRCCompressedSection()
    {
      free(data);
    }
    void write(std::ostream&);
    void prepare();
    uint32_t getSize();

  private:
    virtual void writeData() = 0;
    uint8_t *data;
  protected:
    void compress()
    {
      out.compress();
    }
    bool prepared;
    oPRCFile *parent;
    PRCbitStream out; // order matters: PRCbitStream must be initialized last
};

class PRCGlobalsSection : public PRCCompressedSection
{
  public:
    PRCGlobalsSection(oPRCFile *p, uint32_t i) :
      PRCCompressedSection(p),numberOfReferencedFileStructures(0), 
      tessellationChordHeightRatio(2000.0),tessellationAngleDegrees(40.0),
      defaultFontFamilyName(""),numberOfFonts(0),numberOfPictures(0),
      numberOfTextureDefinitions(0),numberOfFillPatterns(0),
      numberOfReferenceCoordinateSystems(0),userData(0,0),index(i) {}
    uint32_t numberOfReferencedFileStructures;
    double tessellationChordHeightRatio;
    double tessellationAngleDegrees;
    std::string defaultFontFamilyName;
    uint32_t numberOfFonts,numberOfPictures,numberOfTextureDefinitions;
    uint32_t numberOfFillPatterns;
    uint32_t numberOfReferenceCoordinateSystems;
    UserData userData;
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCTreeSection : public PRCCompressedSection
{
  public:
    PRCTreeSection(oPRCFile *p, uint32_t i) :
      PRCCompressedSection(p),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCTessellationSection : public PRCCompressedSection
{
  public:
    PRCTessellationSection(oPRCFile *p, uint32_t i) :
      PRCCompressedSection(p),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCGeometrySection : public PRCCompressedSection
{
  public:
    PRCGeometrySection(oPRCFile *p, uint32_t i) :
      PRCCompressedSection(p),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCExtraGeometrySection : public PRCCompressedSection
{
  public:
    PRCExtraGeometrySection(oPRCFile *p, uint32_t i) :
      PRCCompressedSection(p),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCModelFile : public PRCCompressedSection
{
  public:
    PRCModelFile(oPRCFile *p) : PRCCompressedSection(p) {}
  private:
    virtual void writeData();
};

void makeFileUUID(uint32_t*);
void makeAppUUID(uint32_t*);

class PRCUncompressedFile
{
  public:
    uint32_t file_size;
    uint8_t *data;

    void write(std::ostream&);

    uint32_t getSize();
};

class PRCStartHeader
{
  public:
    uint32_t minimal_version_for_read; // PRCVersion
    uint32_t authoring_version; // PRCVersion
    uint32_t fileStructureUUID[4];
    uint32_t applicationUUID[4]; // should be 0

    void write(std::ostream&);

    uint32_t getSize();
};

class PRCFileStructure
{
  private:
    oPRCFile *parent;
    uint32_t index;
  public:
    PRCStartHeader header;
    uint32_t number_of_uncompressed_files;
    PRCUncompressedFile *uncompressedFiles;

    PRCGlobalsSection globals;
    PRCTreeSection tree;
    PRCTessellationSection tessellations;
    PRCGeometrySection geometry;
    PRCExtraGeometrySection extraGeometry;

    PRCFileStructure(oPRCFile *p, uint32_t i) : parent(p),index(i),
      globals(p,i),tree(p,i),tessellations(p,i),geometry(p,i),
      extraGeometry(p,i) {}
    void write(std::ostream&);
    void prepare();
    uint32_t getSize();
};

class PRCFileStructureInformation
{
  public:
    uint32_t UUID[4];
    uint32_t reserved; // 0
    uint32_t number_of_offsets;
    uint32_t *offsets;

    void write(std::ostream&);

    uint32_t getSize();
};

class PRCHeader
{
  public :
    PRCStartHeader startHeader;
    uint32_t number_of_file_structures;
    PRCFileStructureInformation *fileStructureInformation;
    uint32_t model_file_offset;
    uint32_t file_size; // not documented
    uint32_t number_of_uncompressed_files;
    PRCUncompressedFile *uncompressedFiles;

    void write(std::ostream&);
    uint32_t getSize();
};

class oPRCFile
{
  public:
    oPRCFile(std::ostream &os, uint32_t n=1) :
            number_of_file_structures(n),
            fileStructures(new PRCFileStructure*[n]),modelFile(this),
            fout(NULL),output(os) {
      for(size_t i=0; i < nEntity; ++i)
        count[i]=0;
    }

    oPRCFile(const std::string &name, uint32_t n=1) :
            number_of_file_structures(n),
            fileStructures(new PRCFileStructure*[n]),modelFile(this),
            fout(new std::ofstream(name.c_str())),output(*fout) {
      for(size_t i=0; i < nEntity; ++i)
        count[i]=0;
    }

    ~oPRCFile()
    {
      for(uint32_t i = 0; i < number_of_file_structures; ++i)
        delete fileStructures[i];
      delete[] fileStructures;
      if(fout != NULL)
        delete fout;
    }

    void begingroup(std::string name) {}
    void endgroup() {}
  
    bool add(PRCentity*);
    bool finish();
    uint32_t getColourIndex(const RGBAColour&);
    uint32_t getMaterialIndex(const PRCMaterial&);
    uint32_t getSize();
  
    uint32_t count[nEntity];

    const uint32_t number_of_file_structures;
    PRCFileStructure **fileStructures;
    PRCHeader header;
    PRCModelFile modelFile;
    std::vector<PRCentity*> fileEntities;
    std::vector<RGBAColour> colourMap;
    std::vector<PRCMaterial> materialMap;

  private:
    std::ofstream *fout;
    std::ostream &output;
};

#endif // __O_PRC_FILE_H

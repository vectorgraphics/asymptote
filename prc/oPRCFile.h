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
#include <map>
#include <list>
#include <stack>
#include <string>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "PRC.h"
#include "PRCbitStream.h"
#include "writePRC.h"

class oPRCFile;
class PRCFileStructure;

struct RGBAColour
{
  RGBAColour(double r=0.0, double g=0.0, double b=0.0, double a=1.0) :
    R(r), G(g), B(b), A(a) {}
  double R,G,B,A;

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
};
typedef std::map<RGBAColour,uint32_t> PRCcolourMap;
typedef std::map<PRCRgbColor,uint32_t> PRCcolorMap;

struct PRCmaterial
{
  PRCmaterial() : alpha(1.0),shininess(1.0) {}
  PRCmaterial(const RGBAColour &a, const RGBAColour &d, const RGBAColour &e,
              const RGBAColour &s, double p, double h) :
      ambient(a), diffuse(d), emissive(e), specular(s), alpha(p), shininess(h) {}
  RGBAColour ambient,diffuse,emissive,specular;
  double alpha,shininess;

  bool operator==(const PRCmaterial &m) const
  {
    return (ambient==m.ambient && diffuse==m.diffuse && emissive==m.emissive
        && specular==m.specular && alpha==m.alpha && shininess==m.shininess);
  }
  bool operator<(const PRCmaterial &m) const
  {
    if(ambient!=m.ambient)
      return (ambient<m.ambient);
    if(diffuse!=m.diffuse)
      return (diffuse<m.diffuse);
    if(emissive!=m.emissive)
      return (emissive<m.emissive);
    if(specular!=m.specular)
      return (specular<m.specular);
    if(alpha!=m.alpha)
      return (alpha<m.alpha);
    return (shininess<m.shininess);
  }
};
typedef std::map<PRCmaterial,uint32_t> PRCmaterialMap;

struct PRCtessrectangle // rectangle
{
  PRCVector3d vertices[4];
  uint32_t style;
};
typedef std::vector<PRCtessrectangle> PRCtessrectangleList;

struct PRCtessline // polyline
{
  std::vector<PRCVector3d> point;
  PRCRgbColor color;
};
typedef std::list<PRCtessline> PRCtesslineList;

struct PRCface
{
  uint32_t style;
  bool transparent;
  PRCpGeneralTransformation3d  transform;
  PRCpFace face;
};
typedef std::tr1::shared_ptr <PRCface> PRCpface;
typedef std::vector <PRCface>  PRCfaceList;

struct PRCcompface
{
  uint32_t style;
  bool transparent;
  PRCpCompressedFace face;
};
typedef std::tr1::shared_ptr <PRCcompface> PRCpcompface;
typedef std::vector <PRCcompface>  PRCcompfaceList;

struct PRCwire
{
  uint32_t style;
  PRCpGeneralTransformation3d  transform;
  PRCpCurve curve;
};
typedef std::tr1::shared_ptr <PRCwire> PRCpwire;
typedef std::vector <PRCwire>  PRCwireList;

typedef std::map <uint32_t,std::vector<PRCVector3d> >  PRCpointsetMap;

class PRCoptions
{
public:
  bool closed;   // render the surface as one-sided; may yield faster rendering 
  bool tess;     // use tessellated mesh to store straight patches
  bool do_break;
  bool no_break; // do not render transparent patches as one-faced nodes
  bool ignore;
  
  PRCoptions() : closed(false), tess(false), do_break(true),
                 no_break(false), ignore(false) {}
};

class PRCgroup
{
 public:
  PRCgroup() : compression(0.0) {}
  PRCgroup(const std::string &name) : name(name), compression(0.0) {}
  PRCfaceList       faces;
  PRCcompfaceList   compfaces;
  PRCtessrectangleList  rectangles;
  PRCtesslineList       lines;
  PRCwireList           wires;
  PRCpointsetMap        points;
  std::string name;
  std::list<PRCgroup> groupList;
  PRCpGeneralTransformation3d  transform;
  PRCoptions options;
  double compression;
};
typedef std::list<PRCgroup> PRCgroupList;

class PRCCompressedSection
{
  public:
    PRCCompressedSection(oPRCFile *p) : data(0),prepared(false),parent(p),fileStructure(NULL),
        out(data,0) {}
    PRCCompressedSection(PRCFileStructure *fs) : data(0),prepared(false),parent(NULL),fileStructure(fs),
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
    PRCFileStructure *fileStructure;
    PRCbitStream out; // order matters: PRCbitStream must be initialized last
};

class PRCGlobalsSection : public PRCCompressedSection
{
  public:
    PRCGlobalsSection(PRCFileStructure *fs, uint32_t i) :
      PRCCompressedSection(fs),numberOfReferencedFileStructures(0), 
      tessellationChordHeightRatio(2000.0),tessellationAngleDegrees(40.0),
      defaultFontFamilyName(""),
      numberOfFillPatterns(0),
      userData(0,0),index(i) {}
    uint32_t numberOfReferencedFileStructures;
    double tessellationChordHeightRatio;
    double tessellationAngleDegrees;
    std::string defaultFontFamilyName;
    std::vector<PRCRgbColor> colors;
    std::vector<PRCPicture> pictures;
    PRCTextureDefinitionList texture_definitions;
    PRCMaterialList materials;
    PRCStyleList styles;
    uint32_t numberOfFillPatterns;
    PRCCoordinateSystemList reference_coordinate_systems;
    std::vector<PRCFontKeysSameFont> font_keys_of_font;
    UserData userData;
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCTreeSection : public PRCCompressedSection
{
  public:
    PRCTreeSection(PRCFileStructure *fs, uint32_t i) :
      PRCCompressedSection(fs),unit(1),index(i) {}
  PRCPartDefinitionList part_definitions;
  PRCProductOccurrenceList product_occurrences;
/*
  PRCMarkupList markups;
  PRCAnnotationItemList annotation_entities;
 */
  double unit;
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCTessellationSection : public PRCCompressedSection
{
  public:
    PRCTessellationSection(PRCFileStructure *fs, uint32_t i) :
      PRCCompressedSection(fs),index(i) {}
  PRCTessList tessellations;
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCGeometrySection : public PRCCompressedSection
{
  public:
    PRCGeometrySection(PRCFileStructure *fs, uint32_t i) :
      PRCCompressedSection(fs),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCExtraGeometrySection : public PRCCompressedSection
{
  public:
    PRCExtraGeometrySection(PRCFileStructure *fs, uint32_t i) :
      PRCCompressedSection(fs),index(i) {}
  private:
    uint32_t index;
    virtual void writeData();
};

class PRCModelFile : public PRCCompressedSection
{
  public:
    PRCModelFile(oPRCFile *p) : PRCCompressedSection(p), unit(1) {}
    double unit;
  private:
    virtual void writeData();
};

void makeFileUUID(uint32_t*);
void makeAppUUID(uint32_t*);

class PRCUncompressedFile
{
  public:
    PRCUncompressedFile() : file_size(0), data(NULL) {}
    PRCUncompressedFile(uint32_t fs, uint8_t *d) : file_size(fs), data(d) {}
//    PRCUncompressedFile(uint32_t fs, const std::string &fn) : file_size(fs), data(NULL), file_name(fn) {}
    ~PRCUncompressedFile() { if(data != NULL) delete[] data; }
    uint32_t file_size;
    uint8_t *data;
//    std::string file_name;

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
    std::list<PRCUncompressedFile> uncompressedFiles;
    PRCTopoContextList contexts;

    PRCGlobalsSection globals;
    PRCTreeSection tree;
    PRCTessellationSection tessellations;
    PRCGeometrySection geometry;
    PRCExtraGeometrySection extraGeometry;

    PRCFileStructure(oPRCFile *p, uint32_t i) : parent(p),index(i),
      globals(this,i),tree(this,i),tessellations(this,i),geometry(this,i),
      extraGeometry(this,i) {}
    void write(std::ostream&);
    void prepare();
    uint32_t getSize();
    uint32_t addPicture(EPRCPictureDataFormat format, uint32_t size, const uint8_t *picture, uint32_t width=0, uint32_t height=0, std::string name="");
    uint32_t addTextureDefinition(PRCTextureDefinition *pTextureDefinition);
    uint32_t addRgbColor(const PRCRgbColor &color);
    uint32_t addRgbColorUnique(const PRCRgbColor &color);
    uint32_t addMaterialGeneric(PRCMaterialGeneric *pMaterialGeneric);
    uint32_t addTextureApplication(PRCTextureApplication *pTextureApplication);
    uint32_t addStyle(PRCStyle *pStyle);
    uint32_t addPartDefinition(PRCPartDefinition *pPartDefinition);
    uint32_t addProductOccurrence(PRCProductOccurrence *pProductOccurrence);
    uint32_t addTopoContext(PRCTopoContext *pTopoContext);
    uint32_t add3DTess(PRC3DTess *p3DTess);
    uint32_t add3DWireTess(PRC3DWireTess *p3DWireTess);
/*
    uint32_t addMarkupTess(PRCMarkupTess *pMarkupTess);
    uint32_t addMarkup(PRCMarkup *pMarkup);
    uint32_t addAnnotationItem(PRCAnnotationItem *pAnnotationItem);
 */
    uint32_t addCoordinateSystem(PRCCoordinateSystem *pCoordinateSystem);
    uint32_t addCoordinateSystemUnique(PRCCoordinateSystem *pCoordinateSystem);
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
    std::list<PRCUncompressedFile> uncompressedFiles;

    void write(std::ostream&);
    uint32_t getSize();
};

typedef std::map <PRCGeneralTransformation3d,uint32_t> PRCtransformMap;

class oPRCFile
{
  public:
    oPRCFile(std::ostream &os, double unit=1, uint32_t n=1) :
      number_of_file_structures(n),
      fileStructures(new PRCFileStructure*[n]),modelFile(this),
      fout(NULL),output(os)
      {
        for(uint32_t i = 0; i < number_of_file_structures; ++i)
        {
          fileStructures[i] = new PRCFileStructure(this,i);
          fileStructures[i]->header.minimal_version_for_read = PRCVersion;
          fileStructures[i]->header.authoring_version = PRCVersion;
          makeFileUUID(fileStructures[i]->header.fileStructureUUID);
          makeAppUUID(fileStructures[i]->header.applicationUUID);
          fileStructures[i]->tree.unit = unit;
        }
        modelFile.unit = unit;
      }
  
    oPRCFile(const std::string &name, double unit=1, uint32_t n=1) :
      number_of_file_structures(n),
      fileStructures(new PRCFileStructure*[n]),modelFile(this),
      fout(new std::ofstream(name.c_str(),
                             std::ios::out|std::ios::binary|std::ios::trunc)),
      output(*fout)
      {
        for(uint32_t i = 0; i < number_of_file_structures; ++i)
        {
          fileStructures[i] = new PRCFileStructure(this,i);
          fileStructures[i]->header.minimal_version_for_read = PRCVersion;
          fileStructures[i]->header.authoring_version = PRCVersion;
          makeFileUUID(fileStructures[i]->header.fileStructureUUID);
          makeAppUUID(fileStructures[i]->header.applicationUUID);
          fileStructures[i]->tree.unit = unit;
        }
        modelFile.unit = unit;
      }

    ~oPRCFile()
    {
      for(uint32_t i = 0; i < number_of_file_structures; ++i)
        delete fileStructures[i];
      delete[] fileStructures;
      if(fout != NULL)
        delete fout;
    }

    void begingroup(const char *name, double compress=0.0,
                    PRCoptions *options=NULL, const double t[][4]=NULL);
    void endgroup();
  
    bool finish();
    uint32_t getSize();
  
    const uint32_t number_of_file_structures;
    PRCFileStructure **fileStructures;
    PRCHeader header;
    PRCModelFile modelFile;
    PRCcolorMap colorMap;
    PRCcolourMap colourMap;
    PRCmaterialMap materialMap;
    PRCgroup rootGroup;
    PRCtransformMap transformMap;
    std::stack<PRCgroupList::iterator> currentGroups;
    PRCgroup& findGroup();
    void doGroup(const PRCgroup& group, PRCPartDefinition *parent_part_definition, PRCProductOccurrence *parent_product_occurrence);
    uint32_t addColor(const PRCRgbColor &color);
    uint32_t addColour(const RGBAColour &colour);
    uint32_t addMaterial(const PRCmaterial &material);
    uint32_t addTransform(const PRCpGeneralTransformation3d &transform);
    void addPoint(const double P[3], const RGBAColour &c);
    void addLine(uint32_t n, const double P[][3], const RGBAColour &c);
    void addBezierCurve(uint32_t n, const double cP[][3], const RGBAColour &c);
    void addCurve(uint32_t d, uint32_t n, const double cP[][3], const double *k, const RGBAColour &c, const double w[]);
    void addRectangle(const double P[][3], const PRCmaterial &m);
    void addPatch(const double cP[][3], const PRCmaterial &m);
    void addSurface(uint32_t dU, uint32_t dV, uint32_t nU, uint32_t nV,
     const double cP[][3], const double *kU, const double *kV, const PRCmaterial &m,
     const double w[]);
#define PRCFACETRANSFORM const double origin[3]=NULL, const double x_axis[3]=NULL, const double y_axis[3]=NULL, double scale=1, const double t[][4]=NULL
    void addTube(uint32_t n, const double cP[][3], const double oP[][3], bool straight, const PRCmaterial &m, PRCFACETRANSFORM);
    void addHemisphere(double radius, const PRCmaterial &m, PRCFACETRANSFORM);
    void addSphere(double radius, const PRCmaterial &m, PRCFACETRANSFORM);
    void addDisk(double radius, const PRCmaterial &m, PRCFACETRANSFORM);
    void addCylinder(double radius, double height, const PRCmaterial &m, PRCFACETRANSFORM);
    void addTorus(double major_radius, double minor_radius, double angle1, double angle2, const PRCmaterial &m, PRCFACETRANSFORM);
#undef PRCFACETRANSFORM


    uint32_t addPicture(EPRCPictureDataFormat format, uint32_t size, const uint8_t *picture, uint32_t width=0, uint32_t height=0,
      std::string name="", uint32_t fileStructure=0)
      { return fileStructures[fileStructure]->addPicture(format, size, picture, width, height, name); }
    uint32_t addTextureDefinition(PRCTextureDefinition *pTextureDefinition, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addTextureDefinition(pTextureDefinition);
      }
    uint32_t addTextureApplication(PRCTextureApplication *pTextureApplication, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addTextureApplication(pTextureApplication);
      }
    uint32_t addRgbColor(const PRCRgbColor &color,
       uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addRgbColor(color);
      }
    uint32_t addRgbColorUnique(const PRCRgbColor &color,
       uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addRgbColorUnique(color);
      }
    uint32_t addMaterialGeneric(PRCMaterialGeneric *pMaterialGeneric,
       uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMaterialGeneric(pMaterialGeneric);
      }
    uint32_t addStyle(PRCStyle *pStyle, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addStyle(pStyle);
      }
    uint32_t addPartDefinition(PRCPartDefinition *pPartDefinition, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addPartDefinition(pPartDefinition);
      }
    uint32_t addProductOccurrence(PRCProductOccurrence *pProductOccurrence, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addProductOccurrence(pProductOccurrence);
      }
    uint32_t addTopoContext(PRCTopoContext *pTopoContext, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addTopoContext(pTopoContext);
      }
    uint32_t add3DTess(PRC3DTess *p3DTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->add3DTess(p3DTess);
      }
    uint32_t add3DWireTess(PRC3DWireTess *p3DWireTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->add3DWireTess(p3DWireTess);
      }
/*
    uint32_t addMarkupTess(PRCMarkupTess *pMarkupTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMarkupTess(pMarkupTess);
      }
    uint32_t addMarkup(PRCMarkup *pMarkup, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMarkup(pMarkup);
      }
    uint32_t addAnnotationItem(PRCAnnotationItem *pAnnotationItem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addAnnotationItem(pAnnotationItem);
      }
 */
    uint32_t addCoordinateSystem(PRCCoordinateSystem *pCoordinateSystem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addCoordinateSystem(pCoordinateSystem);
      }
    uint32_t addCoordinateSystemUnique(PRCCoordinateSystem *pCoordinateSystem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addCoordinateSystemUnique(pCoordinateSystem);
      }
  private:
    std::ofstream *fout;
    std::ostream &output;
};

#endif // __O_PRC_FILE_H

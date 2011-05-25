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

struct RGBAColourWidth
{
  RGBAColourWidth(double r=0.0, double g=0.0, double b=0.0, double a=1.0, double w=1.0) :
    R(r), G(g), B(b), A(a), W(w) {}
  double R,G,B,A,W;

  bool operator==(const RGBAColourWidth &c) const
  {
    return (R==c.R && G==c.G && B==c.B && A==c.A && W==c.W);
  }
  bool operator!=(const RGBAColourWidth &c) const
  {
    return !(R==c.R && G==c.G && B==c.B && A==c.A && W==c.W);
  }
  bool operator<(const RGBAColourWidth &c) const
  {
    if(R!=c.R)
      return (R<c.R);
    if(G!=c.G)
      return (G<c.G);
    if(B!=c.B)
      return (B<c.B);
    if(A!=c.A)
      return (A<c.A);
    return (W<c.W);
  }
};
typedef std::map<RGBAColourWidth,uint32_t> PRCcolourwidthMap;

typedef std::map<PRCRgbColor,uint32_t> PRCcolorMap;

struct PRCmaterial
{
  PRCmaterial() : alpha(1.0),shininess(1.0),
      picture(NULL), picture_width(0), picture_height(0), picture_rgba(false), picture_replace(false) {}
  PRCmaterial(const RGBAColour &a, const RGBAColour &d, const RGBAColour &e,
              const RGBAColour &s, double p, double h,
              const uint8_t* pic=NULL, uint32_t picw=0, uint32_t pich=0, bool pica=false, bool picr=false) :
      ambient(a), diffuse(d), emissive(e), specular(s), alpha(p), shininess(h),
      picture(pic), picture_width(picw), picture_height(pich), picture_rgba(pica), picture_replace(picr) {}
  RGBAColour ambient,diffuse,emissive,specular;
  double alpha,shininess;
  const uint8_t* picture;
  uint32_t picture_width;
  uint32_t picture_height;
  bool picture_rgba;    // is there alpha component?
  bool picture_replace; // replace material color with texture color? if false - just modify

  bool operator==(const PRCmaterial &m) const
  {
    return (ambient==m.ambient && diffuse==m.diffuse && emissive==m.emissive
        && specular==m.specular && alpha==m.alpha && shininess==m.shininess
        && picture==m.picture && picture_width==m.picture_width && picture_height==m.picture_height
        && picture_rgba==m.picture_rgba && picture_replace==m.picture_replace);
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
    if(shininess!=m.shininess)
      return (shininess<m.shininess);
    if(picture!=m.picture)
      return (picture<m.picture);
    if(picture_width!=m.picture_width)
      return (picture_width<m.picture_width);
    if(picture_height!=m.picture_height)
      return (picture_height<m.picture_height);
    if(picture_rgba!=m.picture_rgba)
      return (picture_rgba<m.picture_rgba);
    return (picture_replace<m.picture_replace);
  }
};
typedef std::map<PRCmaterial,uint32_t> PRCmaterialMap;

struct PRCtessrectangle // rectangle
{
  PRCVector3d vertices[4];
  uint32_t style;
};
typedef std::vector<PRCtessrectangle> PRCtessrectangleList;

struct PRCtesstriangles // triangle
{
  std::vector<PRCVector3d> vertices;
  std::vector<PRCVector3d> normals;
  std::vector<RGBAColour>  colors;
  uint32_t style;
};
typedef std::vector<PRCtesstriangles> PRCtesstrianglesList;

struct PRCtessline // polyline
{
  std::vector<PRCVector3d> point;
  PRCRgbColor color;
};
typedef std::list<PRCtessline> PRCtesslineList;
typedef std::map<double, PRCtesslineList> PRCtesslineMap;

struct PRCface
{
  PRCface() : transform(NULL), face(NULL) {}
  uint32_t style;
  bool transparent;
  PRCGeneralTransformation3d*  transform;
  PRCFace* face;
};
typedef std::vector <PRCface>  PRCfaceList;

struct PRCcompface
{
  PRCcompface() : face(NULL) {}
  uint32_t style;
  bool transparent;
  PRCCompressedFace* face;
};
typedef std::vector <PRCcompface>  PRCcompfaceList;

struct PRCwire
{
  PRCwire() : style(m1), transform(NULL), curve(NULL) {}
  uint32_t style;
  PRCGeneralTransformation3d*  transform;
  PRCCurve* curve;
};
typedef std::vector <PRCwire>  PRCwireList;

typedef std::map <uint32_t,std::vector<PRCVector3d> >  PRCpointsetMap;

class PRCoptions
{
public:
  double compression;
  double granularity;

  bool closed;   // render the surface as one-sided; may yield faster rendering
  bool tess;     // use tessellated mesh to store straight patches
  bool do_break; //
  bool no_break; // do not render transparent patches as one-faced nodes

  PRCoptions(double compression=0.0, double granularity=0.0, bool closed=false,
             bool tess=false, bool do_break=true, bool no_break=false)
    : compression(compression), granularity(granularity), closed(closed),
      tess(tess), do_break(do_break), no_break(no_break) {}
};

class PRCgroup
{
 public:
  PRCgroup() : transform(NULL) {}
  PRCgroup(const std::string &name) : name(name), transform(NULL) {}
  PRCfaceList       faces;
  PRCcompfaceList   compfaces;
  PRCtessrectangleList  rectangles;
  PRCtesslineMap        lines;
  PRCwireList           wires;
  PRCpointsetMap        points;
  std::vector<PRCPointSet*>      pointsets;
  std::vector<PRCPolyBrepModel*> polymodels;
  std::vector<PRCPolyWire*>      polywires;
  std::string name;
  std::list<PRCgroup> groupList;
  PRCGeneralTransformation3d*  transform;
  PRCoptions options;
};
typedef std::list<PRCgroup> PRCgroupList;

void makeFileUUID(PRCUniqueId&);
void makeAppUUID(PRCUniqueId&);

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

    void write(std::ostream&) const;

    uint32_t getSize() const;
};
typedef std::deque <PRCUncompressedFile*>  PRCUncompressedFileList;

class PRCStartHeader
{
  public:
    uint32_t minimal_version_for_read; // PRCVersion
    uint32_t authoring_version; // PRCVersion
    PRCUniqueId file_structure_uuid;
    PRCUniqueId application_uuid; // should be 0

    PRCStartHeader() :
      minimal_version_for_read(PRCVersion), authoring_version(PRCVersion) {}
    void serializeStartHeader(std::ostream&) const;

    uint32_t getStartHeaderSize() const;
};

class PRCFileStructure : public PRCStartHeader
{
  public:
    uint32_t number_of_referenced_file_structures;
    double tessellation_chord_height_ratio;
    double tessellation_angle_degree;
    std::string default_font_family_name;
    std::vector<PRCRgbColor> colors;
    std::vector<PRCPicture> pictures;
    PRCUncompressedFileList uncompressed_files;
    PRCTextureDefinitionList texture_definitions;
    PRCMaterialList materials;
    PRCStyleList styles;
    PRCCoordinateSystemList reference_coordinate_systems;
    std::vector<PRCFontKeysSameFont> font_keys_of_font;
    PRCPartDefinitionList part_definitions;
    PRCProductOccurrenceList product_occurrences;
//  PRCMarkupList markups;
//  PRCAnnotationItemList annotation_entities;
    double unit;
    PRCTopoContextList contexts;
    PRCTessList tessellations;

    uint32_t sizes[6];
    uint8_t *globals_data;
    PRCbitStream globals_out; // order matters: PRCbitStream must be initialized last
    uint8_t *tree_data;
    PRCbitStream tree_out;
    uint8_t *tessellations_data;
    PRCbitStream tessellations_out;
    uint8_t *geometry_data;
    PRCbitStream geometry_out;
    uint8_t *extraGeometry_data;
    PRCbitStream extraGeometry_out;

    ~PRCFileStructure () {
      for(PRCUncompressedFileList::iterator  it=uncompressed_files.begin();  it!=uncompressed_files.end();  ++it) delete *it;
      for(PRCTextureDefinitionList::iterator it=texture_definitions.begin(); it!=texture_definitions.end(); ++it) delete *it;
      for(PRCMaterialList::iterator          it=materials.begin();           it!=materials.end();           ++it) delete *it;
      for(PRCStyleList::iterator             it=styles.begin();              it!=styles.end();              ++it) delete *it;
      for(PRCTopoContextList::iterator       it=contexts.begin();            it!=contexts.end();            ++it) delete *it;
      for(PRCTessList::iterator              it=tessellations.begin();       it!=tessellations.end();       ++it) delete *it;
      for(PRCPartDefinitionList::iterator    it=part_definitions.begin();    it!=part_definitions.end();    ++it) delete *it;
      for(PRCProductOccurrenceList::iterator it=product_occurrences.begin(); it!=product_occurrences.end(); ++it) delete *it;
      for(PRCCoordinateSystemList::iterator  it=reference_coordinate_systems.begin(); it!=reference_coordinate_systems.end(); it++)
        delete *it;

      free(globals_data);
      free(tree_data);
      free(tessellations_data);
      free(geometry_data);
      free(extraGeometry_data);
    }

    PRCFileStructure() :
      number_of_referenced_file_structures(0),
      tessellation_chord_height_ratio(2000.0),tessellation_angle_degree(40.0),
      default_font_family_name(""),
      unit(1),
      globals_data(NULL),globals_out(globals_data,0),
      tree_data(NULL),tree_out(tree_data,0),
      tessellations_data(NULL),tessellations_out(tessellations_data,0),
      geometry_data(NULL),geometry_out(geometry_data,0),
      extraGeometry_data(NULL),extraGeometry_out(extraGeometry_data,0) {}
    void write(std::ostream&);
    void prepare();
    uint32_t getSize();
    void serializeFileStructureGlobals(PRCbitStream&);
    void serializeFileStructureTree(PRCbitStream&);
    void serializeFileStructureTessellation(PRCbitStream&);
    void serializeFileStructureGeometry(PRCbitStream&);
    void serializeFileStructureExtraGeometry(PRCbitStream&);
    uint32_t addPicture(EPRCPictureDataFormat format, uint32_t size, const uint8_t *picture, uint32_t width=0, uint32_t height=0, std::string name="");
    uint32_t addTextureDefinition(PRCTextureDefinition*& pTextureDefinition);
    uint32_t addRgbColor(const PRCRgbColor &color);
    uint32_t addRgbColorUnique(const PRCRgbColor &color);
    uint32_t addMaterialGeneric(PRCMaterialGeneric*& pMaterialGeneric);
    uint32_t addTextureApplication(PRCTextureApplication*& pTextureApplication);
    uint32_t addStyle(PRCStyle*& pStyle);
    uint32_t addPartDefinition(PRCPartDefinition*& pPartDefinition);
    uint32_t addProductOccurrence(PRCProductOccurrence*& pProductOccurrence);
    uint32_t addTopoContext(PRCTopoContext*& pTopoContext);
    uint32_t getTopoContext(PRCTopoContext*& pTopoContext);
    uint32_t add3DTess(PRC3DTess*& p3DTess);
    uint32_t add3DWireTess(PRC3DWireTess*& p3DWireTess);
/*
    uint32_t addMarkupTess(PRCMarkupTess*& pMarkupTess);
    uint32_t addMarkup(PRCMarkup*& pMarkup);
    uint32_t addAnnotationItem(PRCAnnotationItem*& pAnnotationItem);
 */
    uint32_t addCoordinateSystem(PRCCoordinateSystem*& pCoordinateSystem);
    uint32_t addCoordinateSystemUnique(PRCCoordinateSystem*& pCoordinateSystem);
};

class PRCFileStructureInformation
{
  public:
    PRCUniqueId UUID;
    uint32_t reserved; // 0
    uint32_t number_of_offsets;
    uint32_t *offsets;

    void write(std::ostream&);

    uint32_t getSize();
};

class PRCHeader : public PRCStartHeader
{
  public :
    uint32_t number_of_file_structures;
    PRCFileStructureInformation *fileStructureInformation;
    uint32_t model_file_offset;
    uint32_t file_size; // not documented
    PRCUncompressedFileList uncompressed_files;

    void write(std::ostream&);
    uint32_t getSize();
};

typedef std::map <PRCGeneralTransformation3d,uint32_t> PRCtransformMap;

class oPRCFile
{
  public:
    oPRCFile(std::ostream &os, double u=1, uint32_t n=1) :
      number_of_file_structures(n),
      fileStructures(new PRCFileStructure*[n]),
      unit(u),
      modelFile_data(NULL),modelFile_out(modelFile_data,0),
      fout(NULL),output(os)
      {
        for(uint32_t i = 0; i < number_of_file_structures; ++i)
        {
          fileStructures[i] = new PRCFileStructure();
          fileStructures[i]->minimal_version_for_read = PRCVersion;
          fileStructures[i]->authoring_version = PRCVersion;
          makeFileUUID(fileStructures[i]->file_structure_uuid);
          makeAppUUID(fileStructures[i]->application_uuid);
          fileStructures[i]->unit = u;
        }
      }

    oPRCFile(const std::string &name, double u=1, uint32_t n=1) :
      number_of_file_structures(n),
      fileStructures(new PRCFileStructure*[n]),
      unit(u),
      modelFile_data(NULL),modelFile_out(modelFile_data,0),
      fout(new std::ofstream(name.c_str(),
                             std::ios::out|std::ios::binary|std::ios::trunc)),
      output(*fout)
      {
        for(uint32_t i = 0; i < number_of_file_structures; ++i)
        {
          fileStructures[i] = new PRCFileStructure();
          fileStructures[i]->minimal_version_for_read = PRCVersion;
          fileStructures[i]->authoring_version = PRCVersion;
          makeFileUUID(fileStructures[i]->file_structure_uuid);
          makeAppUUID(fileStructures[i]->application_uuid);
          fileStructures[i]->unit = u;
        }
      }

    ~oPRCFile()
    {
      for(uint32_t i = 0; i < number_of_file_structures; ++i)
        delete fileStructures[i];
      delete[] fileStructures;
      if(fout != NULL)
        delete fout;
      free(modelFile_data);
    }

    void begingroup(const char *name, PRCoptions *options=NULL,
                    const double t[][4]=NULL);
    void endgroup();

    bool finish();
    uint32_t getSize();

    const uint32_t number_of_file_structures;
    PRCFileStructure **fileStructures;
    PRCHeader header;
    PRCUnit unit;
    uint8_t *modelFile_data;
    PRCbitStream modelFile_out; // order matters: PRCbitStream must be initialized last
    PRCcolorMap colorMap;
    PRCcolourMap colourMap;
    PRCcolourwidthMap colourwidthMap;
    PRCmaterialMap materialMap;
    PRCgroup rootGroup;
    PRCtransformMap transformMap;
    std::stack<PRCgroupList::iterator> currentGroups;
    PRCgroup& findGroup();
    void doGroup(PRCgroup& group, PRCPartDefinition *parent_part_definition, PRCProductOccurrence *parent_product_occurrence);
    uint32_t addColor(const PRCRgbColor &color);
    uint32_t addColour(const RGBAColour &colour);
    uint32_t addColourWidth(const RGBAColour &colour, double width);
    uint32_t addMaterial(const PRCmaterial &material);
    uint32_t addTransform(PRCGeneralTransformation3d*& transform);
    void addPoint(const double P[3], const RGBAColour &c, double w=1.0);
    void addPoints(uint32_t n, const double P[][3], const RGBAColour &c, double w=1.0);
    void addLines(uint32_t nP, const double P[][3], uint32_t nI, const uint32_t PI[],
                      const RGBAColour &c, double w, bool segment_color,
                      uint32_t nC, const RGBAColour C[], uint32_t nCI, const uint32_t CI[]);
    void addTriangles(uint32_t nP, const double P[][3], uint32_t nI, const uint32_t PI[][3], const PRCmaterial &m,
                      uint32_t nN, const double N[][3],   const uint32_t NI[][3],
                      uint32_t nT, const double T[][2],   const uint32_t TI[][3],
                      uint32_t nC, const RGBAColour C[],  const uint32_t CI[][3],
                      uint32_t nM, const PRCmaterial M[], const uint32_t MI[]);
    void addQuads(uint32_t nP, const double P[][3], uint32_t nI, const uint32_t PI[][4], const PRCmaterial &m,
                      uint32_t nN, const double N[][3],   const uint32_t NI[][4],
                      uint32_t nT, const double T[][2],   const uint32_t TI[][4],
                      uint32_t nC, const RGBAColour C[],  const uint32_t CI[][4],
                      uint32_t nM, const PRCmaterial M[], const uint32_t MI[]);
    void addLine(uint32_t n, const double P[][3], const RGBAColour &c, double w=1.0);
    void addBezierCurve(uint32_t n, const double cP[][3], const RGBAColour &c);
    void addCurve(uint32_t d, uint32_t n, const double cP[][3], const double *k, const RGBAColour &c, const double w[]);
    void addQuad(const double P[][3], const RGBAColour C[]);

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
    void addCone(double radius, double height, const PRCmaterial &m, PRCFACETRANSFORM);
    void addTorus(double major_radius, double minor_radius, double angle1, double angle2, const PRCmaterial &m, PRCFACETRANSFORM);
#undef PRCFACETRANSFORM


    uint32_t addPicture(EPRCPictureDataFormat format, uint32_t size, const uint8_t *picture, uint32_t width=0, uint32_t height=0,
      std::string name="", uint32_t fileStructure=0)
      { return fileStructures[fileStructure]->addPicture(format, size, picture, width, height, name); }
    uint32_t addTextureDefinition(PRCTextureDefinition*& pTextureDefinition, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addTextureDefinition(pTextureDefinition);
      }
    uint32_t addTextureApplication(PRCTextureApplication*& pTextureApplication, uint32_t fileStructure=0)
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
    uint32_t addMaterialGeneric(PRCMaterialGeneric*& pMaterialGeneric,
       uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMaterialGeneric(pMaterialGeneric);
      }
    uint32_t addStyle(PRCStyle*& pStyle, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addStyle(pStyle);
      }
    uint32_t addPartDefinition(PRCPartDefinition*& pPartDefinition, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addPartDefinition(pPartDefinition);
      }
    uint32_t addProductOccurrence(PRCProductOccurrence*& pProductOccurrence, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addProductOccurrence(pProductOccurrence);
      }
    uint32_t addTopoContext(PRCTopoContext*& pTopoContext, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addTopoContext(pTopoContext);
      }
    uint32_t getTopoContext(PRCTopoContext*& pTopoContext, uint32_t fileStructure=0)
    {
      return fileStructures[fileStructure]->getTopoContext(pTopoContext);
    }
    uint32_t add3DTess(PRC3DTess*& p3DTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->add3DTess(p3DTess);
      }
    uint32_t add3DWireTess(PRC3DWireTess*& p3DWireTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->add3DWireTess(p3DWireTess);
      }
/*
    uint32_t addMarkupTess(PRCMarkupTess*& pMarkupTess, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMarkupTess(pMarkupTess);
      }
    uint32_t addMarkup(PRCMarkup*& pMarkup, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addMarkup(pMarkup);
      }
    uint32_t addAnnotationItem(PRCAnnotationItem*& pAnnotationItem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addAnnotationItem(pAnnotationItem);
      }
 */
    uint32_t addCoordinateSystem(PRCCoordinateSystem*& pCoordinateSystem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addCoordinateSystem(pCoordinateSystem);
      }
    uint32_t addCoordinateSystemUnique(PRCCoordinateSystem*& pCoordinateSystem, uint32_t fileStructure=0)
      {
        return fileStructures[fileStructure]->addCoordinateSystemUnique(pCoordinateSystem);
      }
  private:
    void serializeModelFileData(PRCbitStream&);
    std::ofstream *fout;
    std::ostream &output;
};

#endif // __O_PRC_FILE_H

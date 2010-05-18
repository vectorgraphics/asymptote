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

#ifndef __WRITE_PRC_H
#define __WRITE_PRC_H
#include <string>
#include <vector>
#include <map>
#include <iostream>
#if defined(_MSC_VER)
#include <memory>
#else
#include <tr1/memory>
#endif
#include "PRCbitStream.h"
#include "PRC.h"
#include <float.h>
#include <math.h>

static const uint32_t m1=(uint32_t)-1;

class PRCVector3d
{
public :
 double x;
 double y;
 double z;
 PRCVector3d() {}
 PRCVector3d(double fx, double fy, double fz) :
 x(fx), y(fy), z(fz) {}
 PRCVector3d(const double c[]) :
 x(c?c[0]:0), y(c?c[1]:0), z(c?c[2]:0) {}
 PRCVector3d(const PRCVector3d& sVector3d) :
 x(sVector3d.x), y(sVector3d.y), z(sVector3d.z) {}

 void Set(double fx, double fy, double fz)
 { x = fx; y = fy; z = fz; }
 double Dot(const PRCVector3d & sPt) const
 { return(x*sPt.x)+(y*sPt.y)+(z*sPt.z); }
 double LengthSquared()
 { return(x*x+y*y+z*z); }

 friend PRCVector3d operator + (const PRCVector3d& a, const PRCVector3d& b)
 { return PRCVector3d(a.x+b.x,a.y+b.y,a.z+b.z); }
 friend PRCVector3d operator - (const PRCVector3d& a)
 { return PRCVector3d(-a.x,-a.y,-a.z); }
 friend PRCVector3d operator - (const PRCVector3d& a, const PRCVector3d& b)
 { return PRCVector3d(a.x-b.x,a.y-b.y,a.z-b.z); }
 friend PRCVector3d operator * (const PRCVector3d& a, const double d)
 { return PRCVector3d(a.x*d,a.y*d,a.z*d); }
 friend PRCVector3d operator * (const double d, const PRCVector3d& a)
 { return PRCVector3d(a.x*d,a.y*d,a.z*d); }
 friend PRCVector3d operator / (const PRCVector3d& a, const double d)
 { return PRCVector3d(a.x/d,a.y/d,a.z/d); }
 friend PRCVector3d operator * (const PRCVector3d& a, const PRCVector3d& b)
 { return PRCVector3d((a.y*b.z)-(a.z*b.y), (a.z*b.x)-(a.x*b.z), (a.x*b.y)-(a.y*b.x)); }

 void write(PRCbitStream &out) { out << x << y << z; }
 void serializeVector3d(PRCbitStream &pbs) const { pbs << x << y << z; }
 void serializeVector2d(PRCbitStream &pbs) const { pbs << x << y; }

 double Length();
 bool Normalize();

 bool operator==(const PRCVector3d &v) const
 {
  return x==v.x && y==v.y && z==v.z;
 }
 bool operator!=(const PRCVector3d &v) const
 {
  return !(x==v.x && y==v.y && z==v.z);
 }
 bool operator<(const PRCVector3d &v) const
 {
   if(x!=v.x)
     return (x<v.x);
   if(y!=v.y)
     return (y<v.y);
   return (z<v.z);
 }
 friend std::ostream& operator << (std::ostream& out, const PRCVector3d& v)
 {
   out << "(" << v.x << "," << v.y << "," << v.z << ")";
   return out;
 }
};

class UUID
{
  public:
    UUID(uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3) :
      id0(u0),id1(u1),id2(u2),id3(u3) {}
    void write(PRCbitStream &out)
    {
      out << id0 << id1 << id2 << id3;
    }
  private:
    uint32_t id0,id1,id2,id3;
};

uint32_t makeCADID();
uint32_t makePRCID();
void writeUnit(PRCbitStream &,bool,double);

void writeEmptyMarkups(PRCbitStream&);

class UserData
{
  public:
    UserData(uint32_t s = 0, uint8_t* d = 0) : size(s),data(d) {}
    void write(PRCbitStream&);
  private:
    uint32_t size;
    uint8_t* data;
};

union SingleAttributeData
{
  int32_t integer;
  double real;
  uint32_t time;
  const char *text;
};

union AttributeTitle
{
  const char *text;
  uint32_t integer;
};

class SingleAttribute
{
  public:
    SingleAttribute() {}
    SingleAttribute(bool b,AttributeTitle t,uint32_t y,SingleAttributeData d) : 
      titleIsInteger(b), title(t), type(y), data(d) {}
    void write(PRCbitStream&);
  private:
    bool titleIsInteger;
    AttributeTitle title;
    uint32_t type;
    SingleAttributeData data;
};

class Attribute
{
  public:
    Attribute(bool t,AttributeTitle v, uint32_t s, SingleAttribute* sa) :
      titleIsInteger(t),title(v), sizeOfAttributeKeys(s), singleAttributes(sa)
      {}
    void write(PRCbitStream &);
  private:
    bool titleIsInteger;
    AttributeTitle title;
    uint32_t sizeOfAttributeKeys;
    SingleAttribute *singleAttributes;
};

class Attributes
{
  public:
    Attributes(uint32_t n, Attribute* a) : numberOfAttributes(n), attributes(a)
    {}
    void write(PRCbitStream&);
  private:
    uint32_t numberOfAttributes;
    Attribute *attributes;
};

class ContentPRCBase
{
  public:
    ContentPRCBase(Attributes *a, std::string n="",bool efr = false,
                   uint32_t ci = 0, uint32_t cpi = 0, uint32_t pi = 0) :
      attributes(a),name(n),eligibleForReference(efr),CADID(ci),
      CADPersistentID(cpi),PRCID(pi) {}
    void write(PRCbitStream&);
    uint32_t getPRCID() { return PRCID; }
    Attributes *attributes;
    std::string name;
    bool eligibleForReference;
    uint32_t CADID, CADPersistentID, PRCID;
};

class PRCReferenceUniqueIdentifier
{
public:
  PRCReferenceUniqueIdentifier() :
    type(0), unique_identifier(m1) {}
  void serializeReferenceUniqueIdentifier(PRCbitStream&);
  uint32_t type;
// bool reference_in_same_file_structure;
// PRCUniqueId target_file_structure;
  uint32_t unique_identifier;
};

extern AttributeTitle EMPTY_ATTRIBUTE_TITLE;
extern Attribute EMPTY_ATTRIBUTE;
extern Attributes EMPTY_ATTRIBUTES;
extern ContentPRCBase EMPTY_CONTENTPRCBASE;
extern ContentPRCBase EMPTY_CONTENTPRCBASE_WITH_REFERENCE;

extern std::string currentName;
void writeName(PRCbitStream&,const std::string&);
void resetName();

extern uint32_t current_layer_index;
extern uint32_t current_index_of_line_style;
extern uint16_t current_behaviour_bit_field;

void writeGraphics(PRCbitStream&,uint32_t=m1,uint32_t=m1,uint32_t=1,bool=false);
void resetGraphics();

void resetGraphicsAndName();

struct PRCRgbColor
{
  PRCRgbColor(double r=0.0, double g=0.0, double b=0.0) :
    red(r), green(g), blue(b) {}
  double red,green,blue;
  void serializeRgbColor(PRCbitStream&);

  bool operator==(const PRCRgbColor &c) const
  {
    return (red==c.red && green==c.green && blue==c.blue);
  }
  bool operator!=(const PRCRgbColor &c) const
  {
    return !(red==c.red && green==c.green && blue==c.blue);
  }
  bool operator<(const PRCRgbColor &c) const
  {
    if(red!=c.red)
      return (red<c.red);
    if(green!=c.green)
      return (green<c.green);
    return (blue<c.blue);
  }
};

class PRCPicture : public ContentPRCBase
{
public:
  PRCPicture(std::string n="") :
  ContentPRCBase(&EMPTY_ATTRIBUTES,n), format(KEPRCPicture_PNG), uncompressed_file_index(m1), pixel_width(0), pixel_height(0) {}
  void serializePicture(PRCbitStream&);
  EPRCPictureDataFormat format;
  uint32_t uncompressed_file_index;
  uint32_t pixel_width;
  uint32_t pixel_height;
};

struct PRCVector2d
{
 PRCVector2d() :
 x(0.0), y(0.0) {}
 PRCVector2d(double X, double Y) :
 x(X), y(Y) {}
 void serializeVector2d(PRCbitStream&);
 double x;
 double y;
};

class PRCTextureDefinition : public ContentPRCBase
{
public:
  PRCTextureDefinition(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()), picture_index(m1), texture_mapping_attribute(PRC_TEXTURE_MAPPING_DIFFUSE),
    texture_mapping_attribute_intensity(1.0), texture_mapping_attribute_components(PRC_TEXTURE_MAPPING_COMPONENTS_RGBA),
    texture_function(KEPRCTextureFunction_Modulate), texture_applying_mode(PRC_TEXTURE_APPLYING_MODE_NONE),
    texture_wrapping_mode_S(KEPRCTextureWrappingMode_Unknown), texture_wrapping_mode_T(KEPRCTextureWrappingMode_Unknown) // ,
    // texture_transformation(false), texture_flip_S(false), texture_flip_T(false),
    // behaviour(PRC_TRANSFORMATION_Identity), scale(1.0,1.0), uniform_scale(1.0)
    {}
  void serializeTextureDefinition(PRCbitStream&);
  uint32_t picture_index;
  uint32_t texture_mapping_attribute;
  double texture_mapping_attribute_intensity;
  uint8_t texture_mapping_attribute_components;
  EPRCTextureFunction texture_function;
  uint8_t texture_applying_mode;
  EPRCTextureWrappingMode texture_wrapping_mode_S;
  EPRCTextureWrappingMode texture_wrapping_mode_T;
  // bool texture_transformation;
  // bool texture_flip_S;
  // bool texture_flip_T;
  // uint8_t behaviour;
  // PRCVector2d origin;
  // PRCVector2d X;
  // PRCVector2d Y;
  // PRCVector2d scale;
  // double uniform_scale;
  // double X_homegeneous_coord;
  // double Y_homegeneous_coord;
  // double origin_homegeneous_coord;
};
typedef std::tr1::shared_ptr <PRCTextureDefinition> PRCpTextureDefinition;
typedef std::vector <PRCpTextureDefinition>  PRCTextureDefinitionList;

class PRCMaterial
{
public:
  virtual void serializeMaterial(PRCbitStream&) = 0;
};
typedef std::tr1::shared_ptr <PRCMaterial> PRCpMaterial;
typedef std::vector <PRCpMaterial>  PRCMaterialList;

class PRCMaterialGeneric : public ContentPRCBase, public PRCMaterial
{
public:
  PRCMaterialGeneric(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()),
    ambient(m1), diffuse(m1), emissive(m1), specular(m1), 
    shininess(0.0),
    ambient_alpha(1.0), diffuse_alpha(1.0), emissive_alpha(1.0), specular_alpha(1.0)
    {}
  void serializeMaterialGeneric(PRCbitStream&);
  void serializeMaterial(PRCbitStream &pbs) { serializeMaterialGeneric(pbs); }
  uint32_t picture_index;
  uint32_t ambient;
  uint32_t diffuse;
  uint32_t emissive;
  uint32_t specular;
  double shininess;
  double ambient_alpha;
  double diffuse_alpha;
  double emissive_alpha;
  double specular_alpha;

  bool operator==(const PRCMaterialGeneric &m) const
  {
    return (ambient==m.ambient && diffuse==m.diffuse && emissive==m.emissive && specular==m.specular && shininess==m.shininess &&
            ambient_alpha==m.ambient_alpha && diffuse_alpha==m.diffuse_alpha && emissive_alpha==m.emissive_alpha && specular_alpha==m.specular_alpha);
  }
};

class PRCTextureApplication : public ContentPRCBase, public PRCMaterial
{
public:
  PRCTextureApplication(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()),
    material_generic_index(m1), texture_definition_index(m1),
    next_texture_index(m1), UV_coordinates_index(0)
    {}
  void serializeTextureApplication(PRCbitStream&);
  void serializeMaterial(PRCbitStream &pbs) { serializeTextureApplication(pbs); }
  uint32_t material_generic_index;
  uint32_t texture_definition_index;
  uint32_t next_texture_index;
  uint32_t UV_coordinates_index;
};

class PRCStyle : public ContentPRCBase
{
public:
  PRCStyle(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()), line_width(0.0), is_vpicture(false), line_pattern_vpicture_index(m1),
    is_material(false), color_material_index(m1), is_transparency_defined(false), transparency(255), additional(0)
    {}
  void serializeCategory1LineStyle(PRCbitStream&);
  double line_width;
  bool is_vpicture;
  uint32_t line_pattern_vpicture_index;
  bool is_material;
  uint32_t color_material_index;
  bool is_transparency_defined;
  uint8_t transparency;
  uint8_t additional;
};
typedef std::tr1::shared_ptr <PRCStyle> PRCpStyle;
typedef std::vector <PRCpStyle>  PRCStyleList;

class PRCTessFace
{
public:
  PRCTessFace() :
  start_wire(0), used_entities_flag(0),
  start_triangulated(0), number_of_texture_coordinate_indexes(0), 
  is_rgba(false), behaviour(PRC_GRAPHICS_Show)
  {}
  void serializeTessFace(PRCbitStream&);
  std::vector<uint32_t> line_attributes;
  uint32_t start_wire;			// specifing bounding wire seems not to work as of Acrobat/Reader 9.2
  std::vector<uint32_t> sizes_wire;	// specifing bounding wire seems not to work as of Acrobat/Reader 9.2
  uint32_t used_entities_flag;
  uint32_t start_triangulated;
  std::vector<uint32_t> sizes_triangulated;
  uint32_t number_of_texture_coordinate_indexes;
  bool is_rgba;
  std::vector<uint8_t> rgba_vertices;
  uint32_t behaviour;
};
typedef std::tr1::shared_ptr <PRCTessFace> PRCpTessFace;
typedef std::vector <PRCpTessFace>  PRCTessFaceList;

class PRCContentBaseTessData
{
public:
  PRCContentBaseTessData() :
  is_calculated(false) {}
  void serializeContentBaseTessData(PRCbitStream&);
  bool is_calculated;
  std::vector<double> coordinates;
};

class PRCTess : public PRCContentBaseTessData
{
public:
  virtual void serializeBaseTessData(PRCbitStream &pbs) = 0;
};
typedef std::tr1::shared_ptr <PRCTess> PRCpTess;
typedef std::vector <PRCpTess>  PRCTessList;

class PRC3DTess : public PRCTess
{
public:
  PRC3DTess() :
  has_faces(false), has_loops(false),
  crease_angle(25.8419)
  {}
  void serialize3DTess(PRCbitStream&);
  void serializeBaseTessData(PRCbitStream &pbs) { serialize3DTess(pbs); }
  void addTessFace(PRCTessFace *pTessFace);

  bool has_faces;
  bool has_loops;
  double crease_angle;
  std::vector<double> normal_coordinate;
  std::vector<uint32_t> wire_index;		// specifing bounding wire seems not to work as of Acrobat/Reader 9.2
  std::vector<uint32_t> triangulated_index;
  PRCTessFaceList face_tessellation;
  std::vector<double> texture_coordinate;
};

class PRC3DWireTess : public PRCTess
{
public:
  PRC3DWireTess() :
  is_rgba(false), is_segment_color(false) {}
  void serialize3DWireTess(PRCbitStream&);
  void serializeBaseTessData(PRCbitStream &pbs) { serialize3DWireTess(pbs); }

  bool is_rgba;
  bool is_segment_color;
  std::vector<uint32_t> wire_indexes;
  std::vector<uint8_t> rgba_vertices;
};

class PRCMarkupTess : public PRCTess
{
public:
  PRCMarkupTess() :
  behaviour(0)
  {}
  void serializeMarkupTess(PRCbitStream&);
  void serializeBaseTessData(PRCbitStream &pbs) { serializeMarkupTess(pbs); }

  std::vector<uint32_t> codes;
  std::vector<std::string> texts;
  std::string label;
  uint8_t behaviour;
};

class PRCGraphics
{
public:
  PRCGraphics() : layer_index(m1), index_of_line_style(m1), behaviour_bit_field(PRC_GRAPHICS_Show) {}
  void serializeGraphics(PRCbitStream&);
  void serializeGraphicsForced(PRCbitStream&);
  bool has_graphics() { return (index_of_line_style!=m1 || layer_index!=m1 || behaviour_bit_field!=PRC_GRAPHICS_Show) ; }
  uint32_t layer_index;
  uint32_t index_of_line_style;
  uint16_t behaviour_bit_field;
};
typedef std::tr1::shared_ptr <PRCGraphics> PRCpGraphics;
typedef std::vector <PRCpGraphics>  PRCGraphicsList;

void writeGraphics(PRCbitStream&,const PRCGraphics&,bool=false);

class PRCMarkup: public PRCGraphics, public ContentPRCBase
{
public:
  PRCMarkup(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()),
    type(KEPRCMarkupType_Unknown), sub_type(KEPRCMarkupSubType_Unknown), index_tessellation(m1) {}
  void serializeMarkup(PRCbitStream&);
  EPRCMarkupType type;
  EPRCMarkupSubType sub_type;
// vector<PRCReferenceUniqueIdentifier> linked_items;
// vector<PRCReferenceUniqueIdentifier> leaders;
  uint32_t index_tessellation;
};
typedef std::tr1::shared_ptr <PRCMarkup> PRCpMarkup;
typedef std::vector <PRCpMarkup>  PRCMarkupList;

class PRCAnnotationItem: public PRCGraphics, public ContentPRCBase
{
public:
  PRCAnnotationItem(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()) {}
  void serializeAnnotationItem(PRCbitStream&);
  void serializeAnnotationEntity(PRCbitStream &pbs) { serializeAnnotationItem(pbs); }
  PRCReferenceUniqueIdentifier markup;
};
typedef std::tr1::shared_ptr <PRCAnnotationItem> PRCpAnnotationItem;
typedef std::vector <PRCpAnnotationItem>  PRCAnnotationItemList;

class PRCRepresentationItemContent: public PRCGraphics, public ContentPRCBase
{
public:
  PRCRepresentationItemContent(std::string n="") :
    ContentPRCBase(&EMPTY_ATTRIBUTES,n,true,makeCADID(),0,makePRCID()),
    index_local_coordinate_system(m1), index_tessellation(m1) {}
  void serializeRepresentationItemContent(PRCbitStream&);
  uint32_t index_local_coordinate_system;
  uint32_t index_tessellation;
};

class PRCRepresentationItem : public PRCRepresentationItemContent
{
public:
  PRCRepresentationItem(std::string n="") :
    PRCRepresentationItemContent(n) {}
  virtual void serializeRepresentationItem(PRCbitStream &pbs) = 0;
};
typedef std::tr1::shared_ptr <PRCRepresentationItem> PRCpRepresentationItem;
typedef std::vector <PRCpRepresentationItem>  PRCRepresentationItemList;

class PRCBrepModel : public PRCRepresentationItem
{
public:
  PRCBrepModel(std::string n="") :
    PRCRepresentationItem(n), has_brep_data(true), context_id(m1), body_id(m1), is_closed(false) {}
  void serializeBrepModel(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializeBrepModel(pbs); }
  bool has_brep_data;
  uint32_t context_id;
  uint32_t body_id;
  bool is_closed;
};

class PRCPolyBrepModel : public PRCRepresentationItem
{
public:
  PRCPolyBrepModel(std::string n="") :
    PRCRepresentationItem(n), is_closed(false) {}
  void serializePolyBrepModel(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializePolyBrepModel(pbs); }
  bool is_closed;
};

class PRCPointSet : public PRCRepresentationItem
{
public:
  PRCPointSet(std::string n="") :
    PRCRepresentationItem(n) {}
  void serializePointSet(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializePointSet(pbs); }
  std::vector<PRCVector3d> point;
};
typedef std::tr1::shared_ptr <PRCPointSet> PRCpPointSet;

class PRCWire : public PRCRepresentationItem
{
public:
  PRCWire(std::string n="") :
    PRCRepresentationItem(n), has_wire_body(true), context_id(m1), body_id(m1) {}
  void serializeWire(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializeWire(pbs); }
  bool has_wire_body;
  uint32_t context_id;
  uint32_t body_id;
};

class PRCPolyWire : public PRCRepresentationItem
{
public:
  PRCPolyWire(std::string n="") :
    PRCRepresentationItem(n) {}
  void serializePolyWire(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializePolyWire(pbs); }
};

class PRCSet : public PRCRepresentationItem
{
public:
  PRCSet(std::string n="") :
    PRCRepresentationItem(n) {}
  void serializeSet(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializeSet(pbs); }
  uint32_t addBrepModel(PRCBrepModel *pBrepModel);
  uint32_t addPolyBrepModel(PRCPolyBrepModel *pPolyBrepModel);
  uint32_t addPointSet(PRCPointSet *pPointSet);
  uint32_t addSet(PRCSet *pSet);
  uint32_t addWire(PRCWire *pWire);
  uint32_t addPolyWire(PRCPolyWire *pPolyWire);
  uint32_t addRepresentationItem(PRCRepresentationItem *pRepresentationItem);
  uint32_t addRepresentationItem(PRCpRepresentationItem pRepresentationItem);
  PRCRepresentationItemList elements;
};

class PRCTransformation3d
{
public:
  virtual void serializeTransformation3d(PRCbitStream&) const =0;
};
typedef std::tr1::shared_ptr <PRCTransformation3d> PRCpTransformation3d;
typedef std::vector <PRCpTransformation3d> PRCTransformation3dList;

class PRCGeneralTransformation3d : public PRCTransformation3d
{
public:
  PRCGeneralTransformation3d()
  {
    mat[0][0] = mat[1][1] = mat[2][2] = mat[3][3] = 1.0;
    mat[0][1] = mat[0][2] = mat[0][3] = 0.0;
    mat[1][0] = mat[1][2] = mat[1][3] = 0.0;
    mat[2][0] = mat[2][1] = mat[2][3] = 0.0;
    mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
  }
  PRCGeneralTransformation3d(const double t[][4])
  {
    if(t!=NULL)
      for (size_t i=0;i<4;i++)
        for (size_t j=0;j<4;j++)
          mat[i][j]=t[i][j];
    else
    {
      mat[0][0] = mat[1][1] = mat[2][2] = mat[3][3] = 1.0;
      mat[0][1] = mat[0][2] = mat[0][3] = 0.0;
      mat[1][0] = mat[1][2] = mat[1][3] = 0.0;
      mat[2][0] = mat[2][1] = mat[2][3] = 0.0;
      mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
    }
  }
  PRCGeneralTransformation3d(const double t[][4], double scale)
  {
    if(t!=NULL)
    {
      for (size_t i=0;i<3;i++)
        for (size_t j=0;j<4;j++)
          mat[i][j]=scale*t[i][j];
      for(size_t j= 0;j<4; j++)
        mat[3][j] = scale*t[3][j];
    }
    else
    {
      mat[0][0] = mat[1][1] = mat[2][2] = mat[3][3] = 1.0;
      mat[0][1] = mat[0][2] = mat[0][3] = 0.0;
      mat[1][0] = mat[1][2] = mat[1][3] = 0.0;
      mat[2][0] = mat[2][1] = mat[2][3] = 0.0;
      mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
    }
  }
  
  void serializeGeneralTransformation3d(PRCbitStream&) const;
  void serializeTransformation3d(PRCbitStream& pbs)  const { serializeGeneralTransformation3d(pbs); }
  double mat[4][4];
  bool operator==(const PRCGeneralTransformation3d &t) const
  {
    for (size_t i=0;i<4;i++)
      for (size_t j=0;j<4;j++)
        if(mat[i][j]!=t.mat[i][j])
         return false;
    return true;
  }
  bool operator<(const PRCGeneralTransformation3d &t) const
  {
    for (size_t i=0;i<4;i++)
      for (size_t j=0;j<4;j++)
        if(mat[i][j]!=t.mat[i][j])
        {
          return (mat[i][j]<t.mat[i][j]);
        }
    return false;
  }
  void set(const double t[][4])
  {
    if(t!=NULL) 
      for (size_t i=0;i<4;i++)
        for (size_t j=0;j<4;j++)
          mat[i][j]=t[i][j];
  }
  bool isnotidtransform() const {
    return(
    mat[0][0]!=1 || mat[0][1]!=0 || mat[0][2]!=0 || mat[0][3]!=0 ||
    mat[1][0]!=0 || mat[1][1]!=1 || mat[1][2]!=0 || mat[1][3]!=0 ||
    mat[2][0]!=0 || mat[2][1]!=0 || mat[2][2]!=1 || mat[2][3]!=0 ||
    mat[3][0]!=0 || mat[3][1]!=0 || mat[3][2]!=0 || mat[3][3]!=1 );
  }
  bool isidtransform() const {
    return(
    mat[0][0]==1 && mat[0][1]==0 && mat[0][2]==0 && mat[0][3]==0 &&
    mat[1][0]==0 && mat[1][1]==1 && mat[1][2]==0 && mat[1][3]==0 &&
    mat[2][0]==0 && mat[2][1]==0 && mat[2][2]==1 && mat[2][3]==0 &&
    mat[3][0]==0 && mat[3][1]==0 && mat[3][2]==0 && mat[3][3]==1 );
  }
};
typedef std::tr1::shared_ptr <PRCGeneralTransformation3d> PRCpGeneralTransformation3d;
typedef std::vector <PRCpGeneralTransformation3d> PRCGeneralTransformation3dList;

class PRCCartesianTransformation3d : public PRCTransformation3d
{
public:
  PRCCartesianTransformation3d() :
    behaviour(PRC_TRANSFORMATION_Identity), origin(0.0,0.0,0.0), X(1.0,0.0,0.0), Y(0.0,1.0,0.0), Z(0.0,0.0,1.0),
    scale(1.0,1.0,1.0), uniform_scale(1.0),
    X_homogeneous_coord(0.0), Y_homogeneous_coord(0.0), Z_homogeneous_coord(0.0), origin_homogeneous_coord(1.0) {}
  void serializeCartesianTransformation3d(PRCbitStream& pbs) const;
  void serializeTransformation3d(PRCbitStream& pbs) const { serializeCartesianTransformation3d(pbs); }
  uint8_t behaviour;
  PRCVector3d origin;
  PRCVector3d X;
  PRCVector3d Y;
  PRCVector3d Z;
  PRCVector3d scale;
  double uniform_scale;
  double X_homogeneous_coord;
  double Y_homogeneous_coord;
  double Z_homogeneous_coord;
  double origin_homogeneous_coord;
  bool operator==(const PRCCartesianTransformation3d &t) const
  {
    return behaviour==t.behaviour && origin==t.origin && X==t.X && Y==t.Y && Z==t.Z && scale==t.scale && uniform_scale==t.uniform_scale &&
           X_homogeneous_coord==t.X_homogeneous_coord && Y_homogeneous_coord==t.Y_homogeneous_coord &&
           Z_homogeneous_coord==t.Z_homogeneous_coord && origin_homogeneous_coord==t.origin_homogeneous_coord;
  }
};

class PRCTransformation
{
public:
  PRCTransformation() :
    has_transformation(false), geometry_is_2D(false), behaviour(PRC_TRANSFORMATION_Identity),
    origin(0,0,0), x_axis(1,0,0), y_axis(0,1,0), scale(1) {}
  void serializeTransformation(PRCbitStream&);
  bool has_transformation;
  bool geometry_is_2D;
  uint8_t behaviour;
  PRCVector3d origin;
  PRCVector3d x_axis;
  PRCVector3d y_axis;
  double scale;
};

class PRCCoordinateSystem : public PRCRepresentationItem
{
public:
  PRCCoordinateSystem(std::string n="") :
    PRCRepresentationItem(n) {}
  void serializeCoordinateSystem(PRCbitStream&);
  void serializeRepresentationItem(PRCbitStream &pbs) { serializeCoordinateSystem(pbs); }
  PRCpTransformation3d axis_set;
  bool operator==(const PRCCoordinateSystem &t) const
  {
    if(index_local_coordinate_system!=t.index_local_coordinate_system)
      return false;
    PRCGeneralTransformation3d*       axis_set_general = dynamic_cast<PRCGeneralTransformation3d*>(axis_set.get());
    PRCGeneralTransformation3d*     t_axis_set_general = dynamic_cast<PRCGeneralTransformation3d*>(t.axis_set.get());
    PRCCartesianTransformation3d*   axis_set_cartesian = dynamic_cast<PRCCartesianTransformation3d*>(axis_set.get());
    PRCCartesianTransformation3d* t_axis_set_cartesian = dynamic_cast<PRCCartesianTransformation3d*>(t.axis_set.get());
    if(axis_set_general!=NULL)
      return (t_axis_set_general!=NULL?(*axis_set_general==*t_axis_set_general):false); 
    if(axis_set_cartesian!=NULL)
      return (t_axis_set_cartesian!=NULL?(*axis_set_cartesian==*t_axis_set_cartesian):false); 
    return false;
  }
};
typedef std::tr1::shared_ptr <PRCCoordinateSystem> PRCpCoordinateSystem;
typedef std::vector <PRCpCoordinateSystem>  PRCCoordinateSystemList;

struct PRCFontKey
{
  uint32_t font_size;
  uint8_t  attributes;
};

class PRCFontKeysSameFont
{
public:
  void serializeFontKeysSameFont(PRCbitStream&);
  std::string font_name;
  uint32_t char_set;
  std::vector<PRCFontKey> font_keys;

};

// Topology
class PRCBaseGeometry
{
public:
  PRCBaseGeometry() :
    base_information(false) {}
  PRCBaseGeometry(Attributes *a, std::string n="", uint32_t id = 0) :
    base_information(true),attributes(a),name(n),identifier(id) {}
  void serializeBaseGeometry(PRCbitStream&);
  bool base_information;
  Attributes *attributes;
  std::string name;
  uint32_t identifier;
};

class PRCBoundingBox
{
public:
  PRCBoundingBox() : min(0,0,0),max(0,0,0) {}
  PRCBoundingBox(const PRCVector3d &m1, const PRCVector3d& m2) : min(m1),max(m2) {}
  void serializeBoundingBox(PRCbitStream &pbs);
  PRCVector3d min;
  PRCVector3d max;
};

class PRCDomain
{
public:
  void serializeDomain(PRCbitStream &pbs);
  PRCVector2d min;
  PRCVector2d max;
};

class PRCInterval
{
public:
  PRCInterval() : min(0), max(0) {}
  PRCInterval(double m, double M) : min(m), max(M) {}
  void serializeInterval(PRCbitStream &pbs);
  double min;
  double max;
};

class PRCParameterization
{
public:
  PRCParameterization() : parameterization_coeff_a(1), parameterization_coeff_b(0) {}
  PRCParameterization(double min, double max) : interval(min, max), parameterization_coeff_a(1), parameterization_coeff_b(0) {}
  void serializeParameterization(PRCbitStream &pbs);
  PRCInterval interval;
  double parameterization_coeff_a;
  double parameterization_coeff_b;
};

class PRCUVParameterization
{
public:
  PRCUVParameterization() : swap_uv(false),
    parameterization_on_u_coeff_a(1), parameterization_on_v_coeff_a(1),
    parameterization_on_u_coeff_b(0), parameterization_on_v_coeff_b(0) {}
  void serializeUVParameterization(PRCbitStream &pbs);
  bool swap_uv;
  PRCDomain uv_domain;
  double parameterization_on_u_coeff_a;
  double parameterization_on_v_coeff_a;
  double parameterization_on_u_coeff_b;
  double parameterization_on_v_coeff_b;
};

class PRCControlPoint
{
public:
  PRCControlPoint() :
   x(0), y(0), z(0), w(1) {}
  PRCControlPoint(double X, double Y, double Z=0, double W=1) :
   x(X), y(Y), z(Z), w(W) {}
  PRCControlPoint(const PRCVector3d &v) :
   x(v.x), y(v.y), z(v.z), w(1) {}
  void Set(double fx, double fy, double fz, double fw=1)
   { x = fx; y = fy; z = fz; w = fw; }
  double x;
  double y;
  double z;
  double w;
};

class PRCContentSurface: public PRCBaseGeometry
{
public:
  PRCContentSurface() :
    PRCBaseGeometry(), extend_info(KEPRCExtendTypeNone) {}
  PRCContentSurface(std::string n) :
    PRCBaseGeometry(&EMPTY_ATTRIBUTES,n,makeCADID()),extend_info(KEPRCExtendTypeNone) {} 
  void serializeContentSurface(PRCbitStream&);
  EPRCExtendType extend_info;
};

class PRCSurface : public PRCContentSurface
{
public:
  PRCSurface() :
    PRCContentSurface() {}
  PRCSurface(std::string n) :
    PRCContentSurface(n) {}
  virtual void  serializeSurface(PRCbitStream &pbs) = 0;
};
typedef std::tr1::shared_ptr <PRCSurface> PRCpSurface;
typedef std::vector <PRCpSurface>  PRCSurfaceList;

class PRCNURBSSurface : public PRCSurface
{
public:
  PRCNURBSSurface() :
    PRCSurface(), knot_type(KEPRCKnotTypeUnspecified), surface_form(KEPRCBSplineSurfaceFormUnspecified) {}
  PRCNURBSSurface(std::string n) :
    PRCSurface(n), knot_type(KEPRCKnotTypeUnspecified), surface_form(KEPRCBSplineSurfaceFormUnspecified) {}
  void  serializeNURBSSurface(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeNURBSSurface(pbs); }
  bool is_rational;
  uint32_t degree_in_u;
  uint32_t degree_in_v;
  std::vector<PRCControlPoint> control_point;
  std::vector<double> knot_u;
  std::vector<double> knot_v;
  const EPRCKnotType knot_type;
  const EPRCBSplineSurfaceForm surface_form;
};

class PRCContentCurve: public PRCBaseGeometry
{
public:
  PRCContentCurve() :
    PRCBaseGeometry(), extend_info(KEPRCExtendTypeNone), is_3d(true) {}
  PRCContentCurve(std::string n) :
    PRCBaseGeometry(&EMPTY_ATTRIBUTES,n,makeCADID()),extend_info(KEPRCExtendTypeNone), is_3d(true) {} 
  void serializeContentCurve(PRCbitStream&);
  EPRCExtendType extend_info;
  bool is_3d;
};

class PRCCurve : public PRCContentCurve
{
public:
  PRCCurve() :
    PRCContentCurve() {}
  PRCCurve(std::string n) :
    PRCContentCurve(n) {}
  virtual void  serializeCurve(PRCbitStream &pbs) = 0;
};
typedef std::tr1::shared_ptr <PRCCurve> PRCpCurve;
typedef std::vector <PRCpCurve>  PRCCurveList;

class PRCNURBSCurve : public PRCCurve
{
public:
  PRCNURBSCurve() :
    PRCCurve(), knot_type(KEPRCKnotTypeUnspecified), curve_form(KEPRCBSplineCurveFormUnspecified) {}
  PRCNURBSCurve(std::string n) :
    PRCCurve(n), knot_type(KEPRCKnotTypeUnspecified), curve_form(KEPRCBSplineCurveFormUnspecified) {}
  void  serializeNURBSCurve(PRCbitStream &pbs);
  void  serializeCurve(PRCbitStream &pbs) { serializeNURBSCurve(pbs); }
  bool is_rational;
  uint32_t degree;
  std::vector<PRCControlPoint> control_point;
  std::vector<double> knot;
  const EPRCKnotType knot_type;
  const EPRCBSplineCurveForm curve_form;
};

class PRCPolyLine : public PRCCurve, public PRCTransformation, public PRCParameterization
{
public:
  PRCPolyLine() :
    PRCCurve() {}
  PRCPolyLine(std::string n) :
    PRCCurve(n) {}
  void  serializePolyLine(PRCbitStream &pbs);
  void  serializeCurve(PRCbitStream &pbs) { serializePolyLine(pbs); }
  std::vector<PRCVector3d> point;
};

class PRCCircle : public PRCCurve, public PRCTransformation, public PRCParameterization
{
public:
  PRCCircle() :
    PRCCurve(), PRCParameterization(0,2*M_PI) {}
  PRCCircle(std::string n) :
    PRCCurve(n), PRCParameterization(0,2*M_PI) {}
  void  serializeCircle(PRCbitStream &pbs);
  void  serializeCurve(PRCbitStream &pbs) { serializeCircle(pbs); }
  double radius;
};

class PRCComposite : public PRCCurve, public PRCTransformation, public PRCParameterization
{
public:
  PRCComposite() :
    PRCCurve() {}
  PRCComposite(std::string n) :
    PRCCurve(n) {}
  void  serializeComposite(PRCbitStream &pbs);
  void  serializeCurve(PRCbitStream &pbs) { serializeComposite(pbs); }
  PRCCurveList base_curve;
  std::vector<bool> base_sense;
  bool is_closed;
};

class PRCBlend01 : public PRCSurface, public PRCTransformation, public PRCUVParameterization
{
public:
  PRCBlend01() :
    PRCSurface() {}
  PRCBlend01(std::string n) :
    PRCSurface(n) {}
  void  serializeBlend01(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeBlend01(pbs); }
  void  setCenterCurve(PRCCurve *curve) { center_curve.reset(curve); }
  void  setOriginCurve(PRCCurve *curve) { origin_curve.reset(curve); }
  void  setTangentCurve(PRCCurve *curve) { tangent_curve.reset(curve); }
  PRCpCurve center_curve;
  PRCpCurve origin_curve;
  PRCpCurve tangent_curve;
};

class PRCRuled : public PRCSurface, public PRCTransformation, public PRCUVParameterization
{
public:
  PRCRuled() :
    PRCSurface() {}
  PRCRuled(std::string n) :
    PRCSurface(n) {}
  void  serializeRuled(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeRuled(pbs); }
  void  setFirstCurve(PRCCurve *curve) { first_curve.reset(curve); }
  void  setSecondCurve(PRCCurve *curve) { second_curve.reset(curve); }
  PRCpCurve first_curve;
  PRCpCurve second_curve;
};

class PRCSphere : public PRCSurface, public PRCTransformation, public PRCUVParameterization
{
public:
  PRCSphere() :
    PRCSurface() {}
  PRCSphere(std::string n) :
    PRCSurface(n) {}
  void  serializeSphere(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeSphere(pbs); }
  double radius;
};

class PRCCylinder : public PRCSurface, public PRCTransformation, public PRCUVParameterization
{
public:
  PRCCylinder() :
    PRCSurface() {}
  PRCCylinder(std::string n) :
    PRCSurface(n) {}
  void  serializeCylinder(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeCylinder(pbs); }
  double radius;
};

class PRCTorus : public PRCSurface, public PRCTransformation, public PRCUVParameterization
{
public:
  PRCTorus() :
    PRCSurface() {}
  PRCTorus(std::string n) :
    PRCSurface(n) {}
  void  serializeTorus(PRCbitStream &pbs);
  void  serializeSurface(PRCbitStream &pbs) { serializeTorus(pbs); }
  double major_radius;
  double minor_radius;
};

class PRCBaseTopology
{
public:
  PRCBaseTopology() :
    base_information(false),attributes(NULL),identifier(0) {}
  PRCBaseTopology(Attributes *a, std::string n="", uint32_t id = 0) :
    base_information(true),attributes(a),name(n),identifier(id) {}
  void serializeBaseTopology(PRCbitStream&);
  bool base_information;
  Attributes *attributes;
  std::string name;
  uint32_t identifier;
};

class PRCTopoItem
{
public:
  virtual void serializeTopoItem(PRCbitStream&)=0;
};
typedef std::tr1::shared_ptr <PRCTopoItem> PRCpTopoItem;

class PRCContentBody: public PRCBaseTopology
{
public:
  PRCContentBody() :
    PRCBaseTopology(), behavior(0) {}
  PRCContentBody(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()), behavior(0) {}
  void serializeContentBody(PRCbitStream&);
  uint8_t behavior;
};

class PRCBody : public PRCContentBody, public PRCTopoItem
{
public:
  PRCBody() :
    PRCContentBody(), topo_item_type(PRC_TYPE_ROOT) {}
  PRCBody(uint32_t tit) :
    PRCContentBody(), topo_item_type(tit) {}
  PRCBody(uint32_t tit, std::string n) :
    PRCContentBody(n), topo_item_type(tit) {}
  virtual void serializeBody(PRCbitStream &pbs) = 0;
  void serializeTopoItem(PRCbitStream &pbs) { serializeBody(pbs); }
  uint32_t serialType() { return topo_item_type; }
  virtual double serialTolerance() { return 0; }
  const uint32_t topo_item_type;
};
typedef std::tr1::shared_ptr <PRCBody> PRCpBody;
typedef std::vector <PRCpBody>  PRCBodyList;

class PRCContentWireEdge : public PRCBaseTopology
{
public:
  PRCContentWireEdge() :
    PRCBaseTopology(), has_curve_trim_interval(false) {}
  PRCContentWireEdge(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()), has_curve_trim_interval(false) {} 
  void serializeContentWireEdge(PRCbitStream &pbs);
  void setCurve(PRCCurve *pCurve) { curve_3d.reset(pCurve); }
  void setCurve(PRCpCurve pCurve) { curve_3d = pCurve; }
  PRCpCurve curve_3d;
  bool has_curve_trim_interval;
  PRCInterval curve_trim_interval;
};

class PRCWireEdge : public PRCContentWireEdge, public PRCTopoItem
{
public:
  void serializeWireEdge(PRCbitStream &pbs);
  void serializeTopoItem(PRCbitStream &pbs) { serializeWireEdge(pbs); }
};

class PRCSingleWireBody : public PRCBody
{
public:
  PRCSingleWireBody() :
    PRCBody(PRC_TYPE_TOPO_SingleWireBody) {}
  PRCSingleWireBody(std::string n) :
    PRCBody(PRC_TYPE_TOPO_SingleWireBody, n) {}
  void serializeSingleWireBody(PRCbitStream &pbs);
  void serializeBody(PRCbitStream &pbs) { serializeSingleWireBody(pbs); }
  void setWireEdge(PRCWireEdge *wireEdge);
// void setEdge(const PRCEdge &edge);
  PRCpTopoItem wire_edge;
};

class PRCFace : public PRCBaseTopology, public PRCTopoItem, public PRCGraphics
{
public:
  PRCFace() :
    PRCBaseTopology(), have_surface_trim_domain(false), have_tolerance(false), tolerance(0), number_of_loop(0), outer_loop_index(-1) {}
  PRCFace(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()), have_surface_trim_domain(false), have_tolerance(false), tolerance(0), number_of_loop(0), outer_loop_index(-1) {} 
  void serializeFace(PRCbitStream &pbs);
  void serializeTopoItem(PRCbitStream &pbs) { serializeFace(pbs); }
  void setSurface(PRCSurface *pSurface) { base_surface.reset(pSurface); }
  PRCpSurface base_surface;
  const bool have_surface_trim_domain;
  PRCDomain surface_trim_domain;
  const bool have_tolerance;
  const double tolerance;
  const uint32_t number_of_loop;
  const int32_t outer_loop_index;
// PRCLoopList loop;
};
typedef std::tr1::shared_ptr <PRCFace> PRCpFace;
typedef std::vector <PRCpFace>  PRCFaceList;

class PRCShell : public PRCBaseTopology, public PRCTopoItem
{
public:
  PRCShell() :
    PRCBaseTopology(), shell_is_closed(false) {}
  PRCShell(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()), shell_is_closed(false) {} 
  void serializeShell(PRCbitStream &pbs);
  void serializeTopoItem(PRCbitStream &pbs) { serializeShell(pbs); }
  void addFace(PRCFace *pFace, uint8_t orientation=2);
  void addFace(const PRCpFace &pFace, uint8_t orientation=2);
  bool shell_is_closed;
  PRCFaceList face;
  std::vector<uint8_t> orientation_surface_with_shell;
};
typedef std::tr1::shared_ptr <PRCShell> PRCpShell;
typedef std::vector <PRCpShell>  PRCShellList;

class PRCConnex : public PRCBaseTopology, public PRCTopoItem
{
public:
  PRCConnex() :
    PRCBaseTopology() {}
  PRCConnex(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()) {} 
  void serializeConnex(PRCbitStream &pbs);
  void serializeTopoItem(PRCbitStream &pbs) { serializeConnex(pbs); }
  void addShell(PRCShell *pShell);
  PRCShellList shell;
};
typedef std::tr1::shared_ptr <PRCConnex> PRCpConnex;
typedef std::vector <PRCpConnex>  PRCConnexList;

class PRCBrepData : public PRCBody
{
public:
  PRCBrepData() :
    PRCBody(PRC_TYPE_TOPO_BrepData) {}
  PRCBrepData(std::string n) :
    PRCBody(PRC_TYPE_TOPO_BrepData, n) {}
  void serializeBrepData(PRCbitStream &pbs);
  void serializeBody(PRCbitStream &pbs) { serializeBrepData(pbs); }
  void addConnex(PRCConnex *pConnex);
  PRCConnexList connex;
  PRCBoundingBox bounding_box;
};

// For now - treat just the case of Bezier surfaces cubic 4x4 or linear 2x2
class PRCCompressedFace : public PRCBaseTopology, public PRCGraphics
{
public:
  PRCCompressedFace() :
    PRCBaseTopology(), orientation_surface_with_shell(true), degree(0) {}
  PRCCompressedFace(std::string n) :
    PRCBaseTopology(&EMPTY_ATTRIBUTES,n,makeCADID()), orientation_surface_with_shell(true), degree(0) {} 
  void serializeCompressedFace(PRCbitStream &pbs, double brep_data_compressed_tolerance);
  void serializeContentCompressedFace(PRCbitStream &pbs);
  void serializeCompressedAnaNurbs(PRCbitStream &pbs, double brep_data_compressed_tolerance);
  void serializeCompressedNurbs(PRCbitStream &pbs, double brep_data_compressed_tolerance);
  bool orientation_surface_with_shell;
  uint32_t degree;
  std::vector<PRCVector3d> control_point;
};
typedef std::tr1::shared_ptr <PRCCompressedFace> PRCpCompressedFace;
typedef std::vector <PRCpCompressedFace>  PRCCompressedFaceList;

// For now - treat just the case of one connex/one shell
class PRCCompressedBrepData : public PRCBody
{
public:
  PRCCompressedBrepData() :
    PRCBody(PRC_TYPE_TOPO_BrepDataCompress), serial_tolerance(0), brep_data_compressed_tolerance(0) {}
  PRCCompressedBrepData(std::string n) :
    PRCBody(PRC_TYPE_TOPO_BrepDataCompress, n), serial_tolerance(0), brep_data_compressed_tolerance(0) {}
  void serializeCompressedBrepData(PRCbitStream &pbs);
  void serializeBody(PRCbitStream &pbs) { serializeCompressedBrepData(pbs); }
  void serializeCompressedShell(PRCbitStream &pbs);
  double serialTolerance() { return serial_tolerance; }
  double serial_tolerance;
  double brep_data_compressed_tolerance;
  PRCCompressedFaceList face;
};

class PRCTopoContext : public ContentPRCBase
{
public:
  PRCTopoContext(std::string n="") :
  ContentPRCBase(&EMPTY_ATTRIBUTES,n), behaviour(0), granularity(0), tolerance(0),
   have_smallest_face_thickness(false), smallest_thickness(0), have_scale(false), scale(1) {}
  void serializeTopoContext(PRCbitStream&);
  void serializeContextAndBodies(PRCbitStream&);
  void serializeGeometrySummary(PRCbitStream&);
  void serializeContextGraphics(PRCbitStream&);
  uint32_t addSingleWireBody(PRCSingleWireBody *body);
  uint32_t addBrepData(PRCBrepData *body);
  uint32_t addCompressedBrepData(PRCCompressedBrepData *body);
// type PRC_TYPE_TOPO_Context 
  uint8_t  behaviour;
  double granularity;
  double tolerance;
  bool have_smallest_face_thickness;
  double smallest_thickness;
  bool have_scale;
  double scale;
  PRCBodyList body;
};
typedef std::tr1::shared_ptr <PRCTopoContext> PRCpTopoContext;
typedef std::vector <PRCpTopoContext>  PRCTopoContextList;

#endif //__WRITE_PRC_H

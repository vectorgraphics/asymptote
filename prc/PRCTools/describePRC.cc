/************
*
*   This file is part of a tool for reading 3D content in the PRC format.
*   Copyright (C) 2008 Orest Shardt <shardtor (at) gmail dot com>
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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>

#include "../PRC.h"
#include "describePRC.h"

using std::ostringstream; using std::cout; using std::endl;
using std::hex; using std::dec; using std::string; using std::setw;
using std::setfill;

// describe sections

void describeGlobals(BitByBitData &mData)
{
  mData.tellPosition();
  cout << getIndent() << "--Globals--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructureGlobals))
    return;
  indent();

  describeContentPRCBase(mData,false);
  unsigned int numberOfReferencedFileStructures = mData.readUnsignedInt();
  cout << getIndent() << "numberOfReferencedFileStructures "
      << numberOfReferencedFileStructures << endl;
  indent();
  for(unsigned int i = 0; i < numberOfReferencedFileStructures; ++i)
  {
    describeCompressedUniqueID(mData);
  }
  dedent();

  double tessellation_chord_height_ratio = mData.readDouble();
  cout << getIndent() << "tessellation_chord_height_ratio "
      << tessellation_chord_height_ratio << endl;

  double tessellation_angle_degree = mData.readDouble();
  cout << getIndent() << "tessellation_angle_degree "
      << tessellation_angle_degree << endl;

  string default_font_family_name = mData.readString();
  cout << getIndent() << "default_font_family_name \""
      << default_font_family_name << '\"' << endl;

  unsigned int number_of_fonts = mData.readUnsignedInt();
  cout << getIndent() << "number_of_fonts " << number_of_fonts << endl;

  indent();
  for(unsigned int q = 0; q < number_of_fonts; ++q)
  {
    string font_name = mData.readString();
    cout << getIndent() << "font_name \"" << font_name << '\"' << endl;
    unsigned int char_set = mData.readUnsignedInt();
    cout << getIndent() << "char_set " << char_set << endl;
    unsigned int number_of_font_keys = mData.readUnsignedInt();
    cout << getIndent() << "number_of_font_keys " << number_of_font_keys
        << endl;
    indent();
    for(unsigned int i = 0; i < number_of_font_keys; i++)
    {
      unsigned int font_size = mData.readUnsignedInt() - 1;
      cout << getIndent() << "font_size " << font_size << endl;
      unsigned char attributes = mData.readChar();
      cout << getIndent() << "attributes "
          << static_cast<unsigned int>(attributes) << endl;
    }
    dedent();
  }
  dedent();

  unsigned int number_of_colours = mData.readUnsignedInt();
  cout << getIndent() << "number_of_colours " << number_of_colours << endl;
  indent();
  for(unsigned int i = 0; i < number_of_colours; ++i)
    describeRGBColour(mData);
  dedent();

  unsigned int number_of_pictures = mData.readUnsignedInt();
  cout << getIndent() << "number_of_pictures " << number_of_pictures << endl;
  indent();
  for(unsigned int i=0;i<number_of_pictures;i++)
    describePicture(mData);
  dedent();

  unsigned int number_of_texture_definitions = mData.readUnsignedInt();
  cout << getIndent() << "number_of_texture_definitions "
      << number_of_texture_definitions << endl;
  indent();
  for(unsigned int i=0;i<number_of_texture_definitions;i++)
    describeTextureDefinition(mData);
  dedent();


  unsigned int number_of_materials = mData.readUnsignedInt();
  cout << getIndent() << "number_of_materials " << number_of_materials << endl;
  indent();
  for(unsigned int i=0;i<number_of_materials;i++)
    describeMaterial(mData);
  dedent();

  unsigned int number_of_line_patterns = mData.readUnsignedInt();
  cout << getIndent() << "number_of_line_patterns "
      << number_of_line_patterns << endl;
  indent();
  for(unsigned int i=0;i<number_of_line_patterns;i++)
    describeLinePattern(mData);
  dedent();

  unsigned int number_of_styles = mData.readUnsignedInt();
  cout << getIndent() << "number_of_styles " << number_of_styles << endl;
  indent();
  for(unsigned int i=0;i<number_of_styles;i++)
    describeCategory1LineStyle(mData);
  dedent();

  unsigned int number_of_fill_patterns = mData.readUnsignedInt();
  cout << getIndent() << "number_of_fill_patterns "
      << number_of_fill_patterns << endl;
  indent();
  for(unsigned int i=0;i<number_of_fill_patterns;i++)
    describeFillPattern(mData);
  dedent();

  unsigned int number_of_reference_coordinate_systems = mData.readUnsignedInt();
  cout << getIndent() << "number_of_reference_coordinate_systems "
      << number_of_reference_coordinate_systems << endl;
  indent();
  for(unsigned int i=0;i<number_of_reference_coordinate_systems;i++)
    //NOTE: must be PRC_TYPE_RI_CoordinateSystem
    describeRepresentationItem(mData);
  dedent();

  describeUserData(mData);
  dedent();
  mData.tellPosition();
}

void describeTree(BitByBitData &mData)
{
  mData.tellPosition();
  cout << getIndent() << "--Tree--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructureTree))
    return;
  indent();
  describeContentPRCBase(mData,false);

  unsigned int number_of_part_definitions = mData.readUnsignedInt();
  cout << getIndent() << "number_of_part_definitions "
      << number_of_part_definitions << endl;
  indent();
  for(unsigned int i = 0; i < number_of_part_definitions; ++i)
  {
    describePartDefinition(mData);
  }
  dedent();

  unsigned int number_of_product_occurrences = mData.readUnsignedInt();
  cout << getIndent() << "number_of_product_occurrences "
      << number_of_product_occurrences << endl;
  indent();
  for(unsigned int i = 0; i < number_of_product_occurrences; ++i)
  {
    describeProductOccurrence(mData);
  }
  dedent();

  describeFileStructureInternalData(mData);

  describeUserData(mData);
  dedent();
  mData.tellPosition();
}

void describeTessellation(BitByBitData &mData)
{
  mData.tellPosition();
  cout << getIndent() << "--Tessellation--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructureTessellation))
    return;
  indent();

  describeContentPRCBase(mData,false);

  unsigned int number_of_tessellations = mData.readUnsignedInt();
  cout << getIndent() << "number_of_tessellations "
      << number_of_tessellations << endl;
  indent();
  for(unsigned int i = 0; i < number_of_tessellations; ++i)
  {
    unsigned int type = mData.readUnsignedInt();
    cout << getIndent() << "tessellation type " << type << endl;
    switch(type)
    {
      case PRC_TYPE_TESS_3D:
        describe3DTess(mData);
        break;
      case PRC_TYPE_TESS_3D_Wire:
        describe3DWireTess(mData);
        break;
      case PRC_TYPE_TESS_Markup:
        describe3DMarkupTess(mData);
        break;
      case PRC_TYPE_TESS_3D_Compressed:
        describeHighlyCompressed3DTess(mData);
        break;
      default:
        cout << getIndent() << "Unrecognized tessellation data type "
            << type << endl;
        break;
    }
  }
  dedent();

  describeUserData(mData);

  dedent();
  mData.tellPosition();
}

void describeGeometry(BitByBitData &mData)
{
  mData.tellPosition();
  cout << getIndent() << "--Geometry--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructureGeometry))
    return;
  indent();

  describeContentPRCBase(mData,false);

  unsigned int number_of_topological_contexts = mData.readUnsignedInt();
  cout << getIndent() << "number_of_topological_contexts "
      << number_of_topological_contexts << endl;
  indent();
  for(unsigned int i = 0; i < number_of_topological_contexts; ++i)
  {
    describeTopoContext(mData);
    unsigned int number_of_bodies = mData.readUnsignedInt();
    cout << getIndent() << "number_of_bodies " << number_of_bodies << endl;
    for(unsigned int i = 0; i < number_of_bodies; ++i)
    {
      describeBody(mData);
    }
  }
  dedent();

  describeUserData(mData);
  dedent();
  mData.tellPosition();
}

void describeExtraGeometry(BitByBitData &mData)
{
  mData.tellPosition();
  cout << getIndent() << "--Extra Geometry--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructureExtraGeometry))
    return;
  indent();

  describeContentPRCBase(mData,false);

  unsigned int number_of_contexts = mData.readUnsignedInt();
  cout << getIndent() << "number_of_contexts " << number_of_contexts << endl;
  indent();
  for(unsigned int i = 0; i < number_of_contexts; ++i)
  {
    // geometry summary
    unsigned int number_of_bodies = mData.readUnsignedInt();
    cout << getIndent() << "number_of_bodies " << number_of_bodies << endl;
    indent();
    for(unsigned int j = 0; j < number_of_bodies; ++j)
    {
      unsigned int serial_type = mData.readUnsignedInt();
      cout << getIndent() << "serial_type " << serial_type << endl;
      indent();
      if(isCompressedSerialType(serial_type))
        cout << getIndent() << "serialTolerance " << mData.readDouble();
      dedent();
    }
    dedent();
    // context graphics
    resetCurrentGraphics();
    unsigned int number_of_treat_types = mData.readUnsignedInt();
    cout << getIndent() << "number_of_treat_types "
        << number_of_treat_types << endl;
    indent();
    for(unsigned int i = 0; i < number_of_treat_types; ++i)
    {
      cout << getIndent() << "element_type " << mData.readUnsignedInt() << endl;
      unsigned int number_of_elements = mData.readUnsignedInt();
      cout << getIndent() << "number_of_elements "
          << number_of_elements << endl;
      indent();
      for(unsigned int j = 0; j < number_of_elements; ++j)
      {
        if(mData.readBit())
        {
          describeGraphics(mData);
        }
        else
        {
          cout << getIndent() << "Element has no graphics" << endl;
        }
      }
      dedent();
    }
    dedent();
  }
  dedent();

  describeUserData(mData);
  dedent();
  mData.tellPosition();
}


void describeModelFileData(BitByBitData &mData,
                           unsigned int numberOfFileStructures)
{
  mData.tellPosition();
  cout << getIndent() << "--Model File--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_ModelFile))
    return;
  indent();

  describeContentPRCBase(mData,false);

  describeUnit(mData);
  unsigned int numberOfProductOccurrences = mData.readUnsignedInt();
  cout << getIndent() << "Number of Product Occurrences "
      << numberOfProductOccurrences << endl;
  indent();
  for(unsigned int i = 0; i < numberOfProductOccurrences; ++i)
  {
    describeCompressedUniqueID(mData);
    cout << getIndent() << "index_position + 1 = "
        << mData.readUnsignedInt() << endl;
    cout << getIndent() << "active? " << (mData.readBit()?"yes":"no")
        << endl << endl;
  }
  dedent();
  for(unsigned int i = 0; i < numberOfFileStructures; ++i)
  {
    cout << getIndent() << "File Structure Index in Model File "
        << mData.readUnsignedInt() << endl;
  }

  describeUserData(mData);
  dedent();
  mData.tellPosition();
}

// subsections
// void describe(BitByBitData &mData)
// {
//   
// }

void describeLight(BitByBitData &mData)
{
  unsigned int ID = mData.readUnsignedInt();
  if(ID == PRC_TYPE_GRAPH_AmbientLight)
    cout << getIndent() << "--Ambient Light--" << endl;
  else
    return;

  indent();

  describeContentPRCBase(mData,true);

  cout << getIndent() << "ambient colour index: " << mData.readUnsignedInt()-1 << endl
      << getIndent() << "diffuse colour index: " << mData.readUnsignedInt()-1 << endl
      << getIndent() << "specular colour index: " << mData.readUnsignedInt()-1 << endl;

  describeUserData(mData);
  describeUserData(mData); // why?
  dedent();
}

void describeCamera(BitByBitData &mData)
{
  cout << getIndent() << "--Camera--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_GRAPH_Camera))
    return;
  indent();

  describeContentPRCBase(mData,true);

  cout << getIndent() << (mData.readBit()?"orthographic":"perspective") << endl;
  cout << getIndent() << "Camera Position" << endl;
  describeVector3d(mData);
  cout << getIndent() << "Look At Point" << endl;
  describeVector3d(mData);
  cout << getIndent() << "Up" << endl;
  describeVector3d(mData);
  cout << getIndent() << "X field of view angle (perspective) || X scale (orthographic) "
      << mData.readDouble() << endl;
  cout << getIndent() << "Y field of view angle (perspective) || Y scale (orthographic) "
      << mData.readDouble() << endl;
  cout << getIndent() << "aspect ratio x/y " << mData.readDouble() << endl;
  cout << getIndent() << "near z clipping plane distance from viewer " << mData.readDouble() << endl;
  cout << getIndent() << "far z clipping plane distance from viewer " << mData.readDouble() << endl;
  cout << getIndent() << "zoom factor " << mData.readDouble() << endl;
  dedent();
}

bool describeContentCurve(BitByBitData &mData)
{
  describeBaseGeometry(mData);
  cout << getIndent() << "extend_info " << mData.readUnsignedInt() << endl;
  bool is_3d = mData.readBit();
  cout << getIndent() << "is_3d " << (is_3d?"yes":"no") << endl;
  return is_3d;
}

void describeParameterization(BitByBitData &mData)
{
  cout << getIndent() << "--Parameterization--" << endl;
  indent();
  describeExtent1d(mData);
  cout << getIndent() << "parameterization_coeff_a " << mData.readDouble()
      << endl;
  cout << getIndent() << "parameterization_coeff_b " << mData.readDouble()
      << endl;
  dedent();
}

void describeCurvCircle(BitByBitData &mData)
{
  cout << getIndent() << "--Circle--" << endl;
  indent();

  if(describeContentCurve(mData))
    describeTransformation3d(mData);
  else
    describeTransformation2d(mData);

  describeParameterization(mData);

  cout << getIndent() << "radius " << mData.readDouble() << endl;

  dedent();
}

void describeCurvLine(BitByBitData &mData)
{
  cout << getIndent() << "--Line--" << endl;
  indent();

  if(describeContentCurve(mData))
    describeTransformation3d(mData);
  else
    describeTransformation2d(mData);

  describeParameterization(mData);

  dedent();
}

void describeContentWireEdge(BitByBitData &mData)
{
  cout << getIndent() << "--WireEdge--" << endl;
  indent();

  describeBaseTopology(mData);

  describeObject(mData); //3d_curve

  bool curve_trim_interval = mData.readBit();
  cout << getIndent() << "curve_trim_interval "
      << (curve_trim_interval?"yes":"no") << endl;
  if(curve_trim_interval)
  {
    describeExtent1d(mData);
  }

  dedent();
}

void describeUVParametrization(BitByBitData &mData)
{
  cout << getIndent() << "--UV Parameterization--" << endl;
  indent();

  cout << getIndent() << "swap_uv " << (mData.readBit()?"yes":"no") << endl;
  cout << getIndent() << "Domain" << endl;
  indent(); describeExtent2d(mData); dedent();
  cout << getIndent() << "parameterization_on_u_coeff_a "
      << mData.readDouble() << endl;
  cout << getIndent() << "parameterization_on_v_coeff_a "
      << mData.readDouble() << endl;
  cout << getIndent() << "parameterization_on_u_coeff_b "
      << mData.readDouble() << endl;
  cout << getIndent() << "parameterization_on_v_coeff_b "
      << mData.readDouble() << endl;

  dedent();
}

bool isCompressedSerialType(unsigned int type)
{
  return false; // TODO: actually check the type!!!
}

void describeSurfNURBS(BitByBitData &mData)
{
  cout << getIndent() << "--NURBS surface--" << endl;
  indent();

  describeContentSurface(mData);

  bool is_rational = mData.readBit();
  cout << getIndent() << "is_rational " << (is_rational?"yes":"no") << endl;

  unsigned int degree_in_u = mData.readUnsignedInt();
  cout << getIndent() << "degree_in_u " << degree_in_u << endl;
  unsigned int degree_in_v = mData.readUnsignedInt();
  cout << getIndent() << "degree_in_v " << degree_in_v << endl;

  unsigned int number_of_control_points_in_u = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_control_points_in_u "
      << number_of_control_points_in_u << endl;
  unsigned int number_of_control_points_in_v = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_control_points_in_v "
      << number_of_control_points_in_v << endl;

  unsigned int number_of_knots_in_u = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_knots_in_u "
      << number_of_knots_in_u << endl;
  unsigned int number_of_knots_in_v = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_knots_in_v "
      << number_of_knots_in_v << endl;

  indent();
  for(unsigned int i = 0; i < number_of_control_points_in_u; ++i)
  {
    for(unsigned int j = 0; j < number_of_control_points_in_v; ++j)
    {
      double x = mData.readDouble();
      double y = mData.readDouble();
      double z = mData.readDouble();
      cout << getIndent() << "control point " << i << ' ' << j << ": ("
          << x << ',' << y << ',' << z;

      if(is_rational)
        cout << ',' << mData.readDouble();

      cout << ')' << endl;
    }
  }
  dedent();

  cout << getIndent() << "knots in u" << endl;
  indent();
  for(unsigned int i = 0; i < number_of_knots_in_u; ++i)
    cout << getIndent() << mData.readDouble() << endl;
  dedent();

  cout << getIndent() << "knots in v" << endl;
  indent();
  for(unsigned int i = 0; i < number_of_knots_in_v; ++i)
    cout << getIndent() << mData.readDouble() << endl;
  dedent();

  cout << getIndent() << "knot_type " << mData.readUnsignedInt() << endl;
  cout << getIndent() << "surface_form " << mData.readUnsignedInt() << endl;

  dedent();
}

void describeCurvNURBS(BitByBitData &mData)
{
  cout << getIndent() << "--NURBS curve--" << endl;
  indent();
  describeContentCurve(mData);

  bool is_rational = mData.readBit();
  cout << getIndent() << "is_rational " << (is_rational?"yes":"no") << endl;

  unsigned int degree = mData.readUnsignedInt();
  cout << getIndent() << "degree " << degree << endl;

  unsigned int number_of_control_points = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_control_points "
      << number_of_control_points << endl;

  unsigned int number_of_knots = mData.readUnsignedInt()+1;
  cout << getIndent() << "number_of_knots " << number_of_knots << endl;

  indent();
  for(unsigned int i = 0; i < number_of_control_points; ++i)
  {
    double x = mData.readDouble();
    double y = mData.readDouble();
    double z = mData.readDouble();
    cout << getIndent() << "control point " << i << ": ("
        << x << ',' << y << ',' << z;

    if(is_rational)
      cout << ',' << mData.readDouble();

    cout << ')' << endl;
  }
  dedent();

  cout << getIndent() << "knots" << endl;
  indent();
  for(unsigned int i = 0; i < number_of_knots; ++i)
    cout << getIndent() << mData.readDouble() << endl;
  dedent();

  cout << getIndent() << "knot_type " << mData.readUnsignedInt() << endl;
  cout << getIndent() << "surface_form " << mData.readUnsignedInt() << endl;

  dedent();
}

void describeCurvPolyLine(BitByBitData &mData)
{
  cout << getIndent() << "--PolyLine--" << endl;
  indent();
  describeContentCurve(mData);
  describeTransformation3d(mData);
  describeParameterization(mData);

  unsigned int number_of_points = mData.readUnsignedInt();
  cout << getIndent() << "number_of_points " << number_of_points << endl;
  
  indent();
  for(unsigned int i = 0; i < number_of_points; ++i)
  {
    describeVector3d(mData);
  }
  dedent();

  dedent();
}

void describeSurfCylinder(BitByBitData &mData)
{
  cout << getIndent() << "--Cylinder surface--" << endl;
  indent();

  describeContentSurface(mData);
  describeTransformation3d(mData);
  describeUVParametrization(mData);
  cout << getIndent() << "radius " << mData.readDouble() << endl;

  dedent();
}

void describeSurfPlane(BitByBitData &mData)
{
  cout << getIndent() << "--Plane surface--" << endl;
  indent();
  mData.setShowBits(true);
  describeContentSurface(mData);
  mData.tellPosition();
  //TODO: something is wrong, very wrong!!!
  // For now, all this does is search until the end of the data block,
  // assuming that the default parameterization [-inf,inf]x[-inf,inf] was used
  BitPosition bp = mData.getPosition();
  double nInf1 = mData.readDouble();
  double nInf2 = mData.readDouble();
  double inf1 = mData.readDouble();
  double inf2 = mData.readDouble();
  double one1 = mData.readDouble();
  double one2 = mData.readDouble();
  double zero1 = mData.readDouble();
  double zero2 = mData.readDouble();
  while(!(nInf1 == -12345.0 && nInf2 == -12345.0 && inf1 == 12345.0
          && inf2 == 12345.0 && one1 == 1.0 && one2 == 1.0 && zero1 == 0.0
          && zero2 == 0.0))
  {
    mData.setPosition(bp);
    cout << mData.readBit();
    bp = mData.getPosition();

    nInf1 = mData.readDouble();
    nInf2 = mData.readDouble();
    inf1 = mData.readDouble();
    inf2 = mData.readDouble();
    one1 = mData.readDouble();
    one2 = mData.readDouble();
    zero1 = mData.readDouble();
    zero2 = mData.readDouble();
  }
  cout << endl;
  if(bp.bitIndex == 0)
  {
    bp.bitIndex = 7;
    --bp.byteIndex;
  }
  else
  {
    --bp.bitIndex;
  }
  mData.setPosition(bp);

/*
  // this is what the 8137 docs say it should be
  describeTransformation3d(mData);
  cout << getIndent() << "UV domain" << endl;
  indent();
  cout << getIndent() << "Min: " << endl;
  describeVector2d(mData);
  cout << getIndent() << "Max: " << endl;
  describeVector2d(mData);
  dedent();

  cout << getIndent() << "u coef. a = " << mData.readDouble() << endl;
  cout << getIndent() << "v coef. a = " << mData.readDouble() << endl;
  cout << getIndent() << "u coef. b = " << mData.readDouble() << endl;
  cout << getIndent() << "v coef. b = " << mData.readDouble() << endl;
*/
  mData.setShowBits(false);
  dedent();
}

void describeTopoFace(BitByBitData &mData)
{
  cout << getIndent() << "--Face--" << endl;
  indent();

  describeBaseTopology(mData);

  cout << getIndent() << "base_surface" << endl;
  indent();
  describeObject(mData);
  dedent();

  bool surface_trim_domain = mData.readBit();
  cout << getIndent() << "surface_trim_domain "
      << (surface_trim_domain?"yes":"no") << endl;
  if(surface_trim_domain)
  {
    indent();
    describeExtent2d(mData);
    dedent();
  }

  bool have_tolerance = mData.readBit();
  cout << getIndent() << "have_tolerance "
      << (have_tolerance?"yes":"no") << endl;
  if(have_tolerance)
    cout << getIndent() << "tolerance " << mData.readDouble() << endl;

  unsigned int number_of_loops = mData.readUnsignedInt();
  cout << getIndent() << "number_of_loops " << number_of_loops << endl;
  cout << getIndent() << "outer_loop_index " << mData.readInt() << endl;
  indent();
  for(unsigned int i = 0; i < number_of_loops; ++i)
  {
    describeObject(mData);
  }
  dedent();

  dedent();
}

void describeTopoLoop(BitByBitData &mData)
{
  cout << getIndent() << "--Loop--" << endl;
  indent();

  describeBaseTopology(mData);
  
  cout << getIndent() << "orientation_with_surface "
      << static_cast<unsigned int>(mData.readChar()) << endl;
  unsigned int number_of_coedge = mData.readUnsignedInt();
  cout << getIndent() << "number_of_coedge " << number_of_coedge << endl;
  indent();
  for(unsigned int i = 0; i < number_of_coedge; ++i)
  {
    describeObject(mData);
    cout << getIndent() << "neigh_serial_index "
        << mData.readUnsignedInt() << endl;
  }
  dedent();

  dedent();
}

void describeTopoCoEdge(BitByBitData &mData)
{
  cout << getIndent() << "--CoEdge--" << endl;
  indent();

  describeBaseTopology(mData);

  describeObject(mData); // edge
  describeObject(mData); // uv_curve
  cout << getIndent() << "orientation_with_loop "
      << static_cast<unsigned int>(mData.readChar()) << endl;
  cout << getIndent() << "orientation_uv_with_loop "
      << static_cast<unsigned int>(mData.readChar()) << endl;
  dedent();
}

void describeTopoEdge(BitByBitData &mData)
{
  cout << getIndent() << "--Edge--" << endl;
  indent();

  describeContentWireEdge(mData);

  describeObject(mData); // vertex_start
  describeObject(mData); // vertex_end

  bool have_tolerance = mData.readBit();
  cout << getIndent() << "have_tolerance "
      << (have_tolerance?"yes":"no") << endl;
  if(have_tolerance)
    cout << getIndent() << "tolerance " << mData.readDouble() << endl;
  dedent();
}

void describeTopoUniqueVertex(BitByBitData &mData)
{
  cout << getIndent() << "--Unique Vertex--" << endl;
  indent();

  describeBaseTopology(mData);
  describeVector3d(mData);

  bool have_tolerance = mData.readBit();
  cout << getIndent() << "have_tolerance "
      << (have_tolerance?"yes":"no") << endl;
  if(have_tolerance)
    cout << getIndent() << "tolerance " << mData.readDouble() << endl;

  dedent();
}

void describeTopoConnex(BitByBitData &mData)
{
  cout << getIndent() << "--Connex--" << endl;
  indent();
  describeBaseTopology(mData);
  unsigned int number_of_shells = mData.readUnsignedInt();
  cout << getIndent() << "number_of_shells "
      << number_of_shells << endl;
  indent();
  for(unsigned int i = 0; i < number_of_shells; ++i)
  {
    // NOTE: this does not check if the objects are actually shells!
    describeObject(mData);
  }
  dedent();

  dedent();
}

void describeTopoShell(BitByBitData &mData)
{
  cout << getIndent() << "--Shell--" << endl;
  indent();

  describeBaseTopology(mData);

  cout << getIndent() << "shell_is_closed "
      << (mData.readBit()?"yes":"no") << endl;

  unsigned int number_of_faces = mData.readUnsignedInt();
  cout << getIndent() << "number_of_faces " << number_of_faces << endl;
  for(unsigned int i = 0; i < number_of_faces; ++i)
  {
    // NOTE: this does not check if the objects are actually faces!
    describeObject(mData);
    unsigned char orientation = mData.readChar();
    cout << getIndent() << "orientation_surface_with_shell "
        << static_cast<unsigned int>(orientation) << endl;
  }

  dedent();
}

void describeObject(BitByBitData &mData)
{
  cout << getIndent() << "--Object--" << endl;
  bool already_stored = mData.readBit();
  cout << getIndent() << "already_stored "
      << (already_stored?"yes":"no") << endl;
  if(already_stored) // reverse of documentation?
  {
    cout << getIndent() << "index of stored item "
        << mData.readUnsignedInt() << endl;
  }
  else
  {
    unsigned int type = mData.readUnsignedInt();
    switch(type)
    {
      case PRC_TYPE_ROOT:
        cout << getIndent() << "NULL Object" << endl;
        break;
      // topological items
      case PRC_TYPE_TOPO_Connex:
        describeTopoConnex(mData);
        break;
      case PRC_TYPE_TOPO_Shell:
        describeTopoShell(mData);
        break;
      case PRC_TYPE_TOPO_Face:
        describeTopoFace(mData);
        break;
      case PRC_TYPE_TOPO_Loop:
        describeTopoLoop(mData);
        break;
      case PRC_TYPE_TOPO_CoEdge:
        describeTopoCoEdge(mData);
        break;
      case PRC_TYPE_TOPO_Edge:
        describeTopoEdge(mData);
        break;
      case PRC_TYPE_TOPO_UniqueVertex:
        describeTopoUniqueVertex(mData);
        break;
      case PRC_TYPE_TOPO_WireEdge:
        describeContentWireEdge(mData);
        break;

      // curves
      case PRC_TYPE_CRV_Circle:
        describeCurvCircle(mData);
        break;
      case PRC_TYPE_CRV_NURBS:
        describeCurvNURBS(mData);
        break;
      case PRC_TYPE_CRV_PolyLine:
        describeCurvPolyLine(mData);
        break;
      case PRC_TYPE_CRV_Line:
        describeCurvLine(mData);
        break;
      // surfaces
      case PRC_TYPE_SURF_NURBS:
        describeSurfNURBS(mData);
        break;
      case PRC_TYPE_SURF_Cylinder:
        describeSurfCylinder(mData);
        break;
      case PRC_TYPE_SURF_Plane:
        describeSurfPlane(mData);
        break;

      // topological items
      case PRC_TYPE_TOPO_Item:
      case PRC_TYPE_TOPO_MultipleVertex:


      // curves
      case PRC_TYPE_CRV_Base:
      case PRC_TYPE_CRV_Blend02Boundary:
      case PRC_TYPE_CRV_Composite:
      case PRC_TYPE_CRV_OnSurf:
      case PRC_TYPE_CRV_Ellipse:
      case PRC_TYPE_CRV_Equation:
      case PRC_TYPE_CRV_Helix:
      case PRC_TYPE_CRV_Hyperbola:
      case PRC_TYPE_CRV_Intersection:
      case PRC_TYPE_CRV_Offset:
      case PRC_TYPE_CRV_Parabola:
      case PRC_TYPE_CRV_Transform:

      // surfaces
      case PRC_TYPE_SURF_Base:
      case PRC_TYPE_SURF_Blend01:
      case PRC_TYPE_SURF_Blend02:
      case PRC_TYPE_SURF_Blend03:
      case PRC_TYPE_SURF_Cone:
      case PRC_TYPE_SURF_Cylindrical:
      case PRC_TYPE_SURF_Offset:
      case PRC_TYPE_SURF_Pipe:
      case PRC_TYPE_SURF_Ruled:
      case PRC_TYPE_SURF_Sphere:
      case PRC_TYPE_SURF_Revolution:
      case PRC_TYPE_SURF_Extrusion:
      case PRC_TYPE_SURF_FromCurves:
      case PRC_TYPE_SURF_Torus:
      case PRC_TYPE_SURF_Transform:
      case PRC_TYPE_SURF_Blend04:
        cout << getIndent() << "TODO: Unhandled object of type "
            << type << endl;
        break;
      default:
        cout << getIndent() << "Invalid object of type " << type << endl;
        break;
    }
  }
}

void describeBaseTopology(BitByBitData &mData)
{
  bool base_information = mData.readBit();
  cout << getIndent() << "base_information " <<
      (base_information?"yes":"no") << endl;
  if(base_information)
  {
    describeAttributes(mData);
    describeName(mData);
    cout << getIndent() << "identifier " << mData.readUnsignedInt() << endl;
  }
}

void describeBaseGeometry(BitByBitData &mData)
{
  bool base_information = mData.readBit();
  cout << getIndent() << "base_information " <<
      (base_information?"yes":"no") << endl;
  if(base_information)
  {
    describeAttributes(mData);
    describeName(mData);
    cout << getIndent() << "identifier " << mData.readUnsignedInt() << endl;
  }
}

unsigned int describeContentBody(BitByBitData &mData)
{
  describeBaseTopology(mData);
  unsigned int behaviour = static_cast<unsigned int>(mData.readChar());
  cout << getIndent() << "behaviour " << behaviour << endl;
  return behaviour;
}

void describeContentSurface(BitByBitData &mData)
{
  describeBaseGeometry(mData);
  cout << getIndent() << "extend_info " << mData.readUnsignedInt() << endl;
}

void describeBody(BitByBitData &mData)
{
  cout << getIndent() << "--Body--" << endl;
  unsigned int type = mData.readUnsignedInt();
  switch(type)
  {
    case PRC_TYPE_TOPO_BrepData:
    {
      cout << getIndent() << "--PRC_TYPE_TOPO_BrepData--" << endl;
      unsigned int behaviour = describeContentBody(mData);

      unsigned int number_of_connex = mData.readUnsignedInt();
      cout << getIndent() << "number_of_connex " << number_of_connex << endl;
      indent();
      for(unsigned int i = 0; i < number_of_connex; ++i)
      {
        describeObject(mData);
      }
      dedent();
      if(behaviour != 0)
      {
        cout << getIndent() << "bbox " << endl;
        indent();
        describeExtent3d(mData);
        dedent();
      }
      break;
    }
    case PRC_TYPE_TOPO_SingleWireBody:
    {
      cout << getIndent() << "--PRC_TYPE_TOPO_SingleWireBody--" << endl;
      // unsigned int behaviour = describeContentBody(mData);
      // TODO: is behaviour needed to get data about how to describe?
      describeContentBody(mData);

      describeObject(mData); // curve

      break;
    } 
    case PRC_TYPE_TOPO_BrepDataCompress:
    case PRC_TYPE_TOPO_SingleWireBodyCompress:
      cout << getIndent() << "TODO: Unhandled body type " << type << endl;
      break;
    default:
      cout << getIndent() << "Invalid body type " << type << endl;
      break;
  }
}

void describeTopoContext(BitByBitData &mData)
{
  cout << getIndent() << "--Topological Context--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_TOPO_Context))
    return;
  indent();

  describeContentPRCBase(mData,false);

  cout << getIndent() << "behaviour "
      << static_cast<unsigned int>(mData.readChar()) << endl;
  cout << getIndent() << "granularity " << mData.readDouble() << endl;
  cout << getIndent() << "tolerance " << mData.readDouble() << endl;

  bool have_smallest_face_thickness = mData.readBit();
  cout << getIndent() << "have_smallest_face_thickness "
      << (have_smallest_face_thickness?"yes":"no") << endl;
  if(have_smallest_face_thickness)
    cout << getIndent() << "smallest_thickness " << mData.readDouble() << endl;

  bool have_scale = mData.readBit();
  cout << getIndent() << "have_scale " << (have_scale?"yes":"no") << endl;
  if(have_scale)
    cout << getIndent() << "scale " << mData.readDouble() << endl;

  dedent();
}

void describeLineAttr(BitByBitData& mData)
{
  cout << getIndent() << "index_of_line_style "
      << mData.readUnsignedInt()-1 << endl;
}

void describeArrayRGBA(BitByBitData& mData, int number_of_colours,
                       int number_by_vector)
{
  // bool new_colour = true; // not currently used
  for(int i = 0; i < number_by_vector; ++i)
  {
    cout << getIndent() << static_cast<unsigned int>(mData.readChar()) << ' ';
    cout << static_cast<unsigned int>(mData.readChar()) << ' ';
    cout << static_cast<unsigned int>(mData.readChar()) << endl;
    //TODO: finish this
  }
}

void describeContentBaseTessData(BitByBitData &mData)
{
  cout << getIndent() << "is_calculated "
      << (mData.readBit()?"yes":"no") << endl;
  unsigned int number_of_coordinates = mData.readUnsignedInt();
  cout << getIndent() << "number_of_coordinates "
      << number_of_coordinates << endl;
  indent();
  for(unsigned int i = 0; i < number_of_coordinates; ++i)
  {
    cout << getIndent() << mData.readDouble() << endl;
  }
  dedent();
}

void describeTessFace(BitByBitData &mData)
{
  cout << getIndent() << "--Tessellation Face--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_TESS_Face))
    return;
  indent();

  unsigned int size_of_line_attributes = mData.readUnsignedInt();
  cout << getIndent() << "size_of_line_attributes "
      << size_of_line_attributes << endl;
  indent();
  for(unsigned int i = 0; i < size_of_line_attributes; ++i)
  {
    describeLineAttr(mData);
  }
  dedent();

  unsigned int start_wire = mData.readUnsignedInt();
  cout << getIndent() << "start_wire " << start_wire << endl;
  unsigned int size_of_sizes_wire = mData.readUnsignedInt();
  cout << getIndent() << "size_of_sizes_wire " << size_of_sizes_wire << endl;
  indent();
  for(unsigned int i = 0; i < size_of_sizes_wire; ++i)
  {
    cout << getIndent() << mData.readUnsignedInt() << endl;
  }
  dedent();


  unsigned int used_entities_flag = mData.readUnsignedInt();
  cout << getIndent() << "used_entities_flag " << used_entities_flag << endl;

  unsigned int start_triangulated = mData.readUnsignedInt();
  cout << getIndent() << "start_triangulated " << start_triangulated << endl;
  unsigned int size_of_sizes_triangulated = mData.readUnsignedInt();
  cout << getIndent() << "size_of_sizes_triangulated "
      << size_of_sizes_triangulated << endl;
  indent();
  for(unsigned int i = 0; i < size_of_sizes_triangulated; ++i)
  {
    cout << getIndent() << mData.readUnsignedInt() << endl;
  }
  dedent();

  cout << getIndent() << "number_of_texture_coordinate_indexes "
      << mData.readUnsignedInt() << endl;

  bool has_vertex_colors = mData.readBit();
  cout << getIndent() << "has_vertex_colors "
      << (has_vertex_colors?"yes":"no") << endl;
  indent();
  if(has_vertex_colors)
  {
    bool is_rgba = mData.readBit();
    cout << getIndent() << "is_rgba " << (is_rgba?"yes":"no") << endl;

    bool b_optimised = mData.readBit();
    cout << getIndent() << "b_optimised " << (b_optimised?"yes":"no") << endl;
    if(!b_optimised)
    {
      indent();
      //TODO: compute size of Array and pass it instead of 0
      describeArrayRGBA(mData,0,(is_rgba ? 4 : 3));
      dedent();
    }
    else
    {
      // not described
      // what does this mean? that this should not happen?
      // or is omitted from the docs?
    }
  }
  dedent();

  if(size_of_line_attributes)
  {
    cout << getIndent() << "behaviour " << mData.readUnsignedInt() << endl;
  }

  dedent();
}

void describe3DTess(BitByBitData &mData)
{
  cout << getIndent() << "--3D Tessellation--" << endl;
  indent();

  describeContentBaseTessData(mData);

  cout << getIndent() << "has_faces " << (mData.readBit()?"yes":"no") << endl;
  cout << getIndent() << "has_loops " << (mData.readBit()?"yes":"no") << endl;

  bool must_recalculate_normals = mData.readBit();
  cout << getIndent() << "must_recalculate_normals "
      << (must_recalculate_normals?"yes":"no") << endl;
  indent();
  if(must_recalculate_normals)
  {
    cout << getIndent()
        << "Docs were wrong: must_recalculate_normals is true." << endl;
    cout << getIndent() << "normals_recalculation_flags "
        << static_cast<unsigned int>(mData.readChar()) << endl;
    cout << getIndent() << "crease_angle " << mData.readDouble() << endl;
  }
  dedent();

  unsigned int number_of_normal_coordinates = mData.readUnsignedInt();
  cout << getIndent() << "number_of_normal_coordinates "
      << number_of_normal_coordinates << endl;
  indent();
  for(unsigned int i = 0; i < number_of_normal_coordinates; ++i)
  {
    cout << getIndent() << mData.readDouble() << endl;
  }
  dedent();

  unsigned int number_of_wire_indices = mData.readUnsignedInt();
  cout << getIndent() << "number_of_wire_indices "
      << number_of_wire_indices << endl;
  indent();
  for(unsigned int i = 0; i < number_of_wire_indices; ++i)
  {
    cout << getIndent() << mData.readUnsignedInt() << endl;
  }
  dedent();

  unsigned int number_of_triangulated_indices = mData.readUnsignedInt();
  cout << getIndent() << "number_of_triangulated_indices "
      << number_of_triangulated_indices << endl;
  indent();
  for(unsigned int i = 0; i < number_of_triangulated_indices; ++i)
  {
    cout << getIndent() << mData.readUnsignedInt() << endl;
  }
  dedent();

  unsigned int number_of_face_tessellation = mData.readUnsignedInt();
  cout << getIndent() << "number_of_face_tessellation "
      << number_of_face_tessellation << endl;
  indent();
  for(unsigned int i = 0; i < number_of_face_tessellation; ++i)
  {
    describeTessFace(mData);
  }
  dedent();

  unsigned int number_of_texture_coordinates = mData.readUnsignedInt();
  cout << getIndent() << "number_of_texture_coordinates "
      << number_of_texture_coordinates << endl;
  indent();
  for(unsigned int i = 0; i < number_of_texture_coordinates; ++i)
  {
    cout << getIndent() << mData.readDouble() << endl;
  }
  dedent();

  dedent();
}

void describe3DWireTess(BitByBitData &mData)
{
  //TODO
}

void describe3DMarkupTess(BitByBitData &mData)
{
  //TODO
}

void describeHighlyCompressed3DTess(BitByBitData &mData)
{
  //TODO
}

void describeSceneDisplayParameters(BitByBitData &mData)
{
  cout << getIndent() << "--Scene Display Parameters--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_GRAPH_SceneDisplayParameters))
    return;
  indent();
  describeContentPRCBase(mData,true);

  cout << getIndent() << "is active? " << (mData.readBit()?"yes":"no") << endl;

  unsigned int number_of_lights = mData.readUnsignedInt();
  cout << getIndent() << "number of lights " << number_of_lights << endl;
  indent();
  for(unsigned int i = 0; i < number_of_lights; ++i)
  {
    describeLight(mData);
  }
  dedent();

  bool camera = mData.readBit();
  cout << getIndent() << "camera? " << (camera?"yes":"no") << endl;
  if(camera)
    describeCamera(mData);

  bool rotation_centre = mData.readBit();
  cout << getIndent() << "rotation centre? " << (rotation_centre?"yes":"no") << endl;
  if(rotation_centre)
    describeVector3d(mData);

  unsigned int number_of_clipping_planes = mData.readUnsignedInt();
  cout << getIndent() << "number of clipping planes " << number_of_clipping_planes << endl;
  indent();
  for(unsigned int i = 0; i < number_of_clipping_planes; ++i)
  {
    cout << "Can't describe planes!!!" << endl;
    //describePlane(mData);
  }
  dedent();

  cout << getIndent() << "Background line style index: " << mData.readUnsignedInt()-1 << endl;
  cout << getIndent() << "Default line style index: " << mData.readUnsignedInt()-1 << endl;

  unsigned int number_of_default_styles_per_type = mData.readUnsignedInt();
  cout << getIndent() << "number_of_default_styles_per_type " << number_of_default_styles_per_type << endl;
  indent();
  for(unsigned int i = 0; i < number_of_default_styles_per_type; ++i)
  {
    cout << getIndent() << "type " << mData.readUnsignedInt() << endl;
    cout << getIndent() << "line style index: " << mData.readUnsignedInt()-1 << endl;
  }
  dedent();

  dedent();
}

void describeCartesionTransformation3d(BitByBitData& mData)
{
  cout << getIndent() << "--3d Cartesian Transformation--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_MISC_CartesianTransformation))
    return;
  indent();
  unsigned char behaviour = mData.readChar();
  cout << getIndent() << "behaviour "
      << static_cast<unsigned int>(behaviour) << endl;
  if((behaviour & PRC_TRANSFORMATION_Translate) != 0)
  {
    cout << getIndent() << "Translation" << endl;
    describeVector3d(mData);
  }

  if((behaviour & PRC_TRANSFORMATION_NonOrtho) != 0)
  {
    cout << getIndent() << "Non orthogonal transformation" << endl;
    cout << getIndent() << "X" << endl; describeVector3d(mData);
    cout << getIndent() << "Y" << endl; describeVector3d(mData);
    cout << getIndent() << "Z" << endl; describeVector3d(mData);
  }
  else if((behaviour & PRC_TRANSFORMATION_Rotate) != 0)
  {
    cout << getIndent() << "Rotation" << endl;
    cout << getIndent() << "X" << endl; describeVector3d(mData);
    cout << getIndent() << "Y" << endl; describeVector3d(mData);
  }

  // this is different from the docs!!! but it works...
  if ((behaviour & PRC_TRANSFORMATION_NonUniformScale) != 0)
  {
    cout << getIndent() << "Non-uniform scale by " << endl;
    describeVector3d(mData);
  }

  // this is different from the docs!!! but it works...
  if((behaviour & PRC_TRANSFORMATION_Scale) != 0)
  {
    cout << getIndent() << "Uniform Scale by " << mData.readDouble() << endl;
  }

  if((behaviour & PRC_TRANSFORMATION_Homogeneous) != 0)
  {
    cout << getIndent() << "transformation has homogenous values" << endl;
    cout << getIndent() << "x = " << mData.readDouble() << endl;
    cout << getIndent() << "y = " << mData.readDouble() << endl;
    cout << getIndent() << "z = " << mData.readDouble() << endl;
    cout << getIndent() << "w = " << mData.readDouble() << endl;
  }
  dedent();
}

void describeTransformation3d(BitByBitData& mData)
{
  cout << getIndent() << "--3d Transformation--" << endl;
  indent();
  bool has_transformation = mData.readBit();
  cout << getIndent() << "has_transformation "
      << (has_transformation?"yes":"no") << endl;
  if(has_transformation)
  {
    unsigned char behaviour = mData.readChar();
    cout << getIndent() << "behaviour "
        << static_cast<unsigned int>(behaviour) << endl;
    if((behaviour & PRC_TRANSFORMATION_Translate) != 0)
    {
      cout << getIndent() << "Translation" << endl;
      describeVector3d(mData);
    }
    if((behaviour & PRC_TRANSFORMATION_Rotate) != 0)
    {
      cout << getIndent() << "Rotation" << endl;
      cout << getIndent() << "X" << endl; describeVector3d(mData);
      cout << getIndent() << "Y" << endl; describeVector3d(mData);
    }

    if((behaviour & PRC_TRANSFORMATION_Scale) != 0)
    {
      cout << getIndent() << "Uniform Scale by " << mData.readDouble() << endl;
    }
  }
  dedent();
}

void describeTransformation2d(BitByBitData& mData)
{
  cout << getIndent() << "--2d Transformation--" << endl;
  indent();
  bool has_transformation = mData.readBit();
  cout << "has_transformation " << (has_transformation?"yes":"no") << endl;
  if(has_transformation)
  {
    unsigned char behaviour = mData.readChar();
    cout << getIndent() << "behaviour "
        << static_cast<unsigned int>(behaviour) << endl;
    if((behaviour & PRC_TRANSFORMATION_Translate) != 0)
    {
      cout << getIndent() << "Translation" << endl;
      describeVector2d(mData);
    }
    if((behaviour & PRC_TRANSFORMATION_Rotate) != 0)
    {
      cout << getIndent() << "Rotation" << endl;
      cout << getIndent() << "X" << endl; describeVector2d(mData);
      cout << getIndent() << "Y" << endl; describeVector2d(mData);
    }

    if((behaviour & PRC_TRANSFORMATION_Scale) != 0)
    {
      cout << getIndent() << "Uniform Scale by " << mData.readDouble() << endl;
    }
  }
  dedent();
}

void describeFileStructureInternalData(BitByBitData &mData)
{
  cout << getIndent() << "--File Structure Internal Data--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_FileStructure))
    return;
  indent();
  describeContentPRCBase(mData,false);
  cout << getIndent() << "next_available_index "
      << mData.readUnsignedInt() << endl;
  cout << getIndent() << "index_product_occurence "
      << mData.readUnsignedInt() << endl;
  dedent();
}

void describeProductOccurrence(BitByBitData &mData)
{
  cout << getIndent() << "--Product Occurrence--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_ProductOccurence))
    return;
  indent();

  describeContentPRCBaseWithGraphics(mData,true);

  cout << getIndent() << "index_part "
      << static_cast<int>(mData.readUnsignedInt()-1) << endl;
  unsigned int index_prototype = mData.readUnsignedInt()-1;
  cout << getIndent() << "index_prototype "
      << static_cast<int>(index_prototype) << endl;
  if(index_prototype+1 != 0)
  {
    bool prototype_in_same_file_structure = mData.readBit();
    cout << getIndent() << "prototype_in_same_file_structure "
        << (prototype_in_same_file_structure?"yes":"no") << endl;
    if(!prototype_in_same_file_structure)
      describeCompressedUniqueID(mData);
  }

  unsigned int index_external_data = mData.readUnsignedInt()-1;
  cout << getIndent() << "index_external_data "
      << static_cast<int>(index_external_data) << endl;
  if(index_external_data+1 != 0)
  {
    bool external_data_in_same_file_structure = mData.readBit();
    cout << getIndent() << "external_data_in_same_file_structure "
        << (external_data_in_same_file_structure?"yes":"no") << endl;
    if(!external_data_in_same_file_structure)
      describeCompressedUniqueID(mData);
  }

  unsigned int number_of_son_product_occurences = mData.readUnsignedInt();
  cout << getIndent() << "number_of_son_product_occurences "
      << number_of_son_product_occurences << endl;
  indent();
  for(unsigned int i = 0; i < number_of_son_product_occurences; ++i)
    cout << getIndent() << mData.readUnsignedInt() << endl;
  dedent();

  cout << getIndent() << "product_behaviour "
      << static_cast<unsigned int>(mData.readChar()) << endl;

  describeUnit(mData);
  cout << getIndent() << "Product information flags "
      << static_cast<unsigned int>(mData.readChar()) << endl;
  cout << getIndent() << "product_load_status "
      << mData.readUnsignedInt() << endl;

  bool has_location = mData.readBit();
  cout << getIndent() << "has_location " << has_location << endl;
  if(has_location)
  {
    describeCartesionTransformation3d(mData);
  }

  unsigned int number_of_references = mData.readUnsignedInt();
  cout << getIndent() << "number_of_references "
      << number_of_references << endl;
  indent();
  for(unsigned int i = 0; i < number_of_references; ++i)
  {
    //TODO: describeReferenceToPRCBase(mData);
  }
  dedent();

  describeMarkups(mData);

  unsigned int number_of_views = mData.readUnsignedInt();
  cout << getIndent() << "number_of_views " << number_of_views << endl;
  indent();
  for(unsigned int i = 0; i < number_of_views; ++i)
  {
    describeAnnotationView(mData);
  }
  dedent();

  bool has_entity_filter = mData.readBit();
  cout << getIndent() << "has_entity_filter "
      << (has_entity_filter?"yes":"no") << endl;
  if(has_entity_filter)
  {
    //TODO: describeEntityFilter(mData);
  }

  unsigned int number_of_display_filters = mData.readUnsignedInt();
  cout << getIndent() << "number_of_display_filters "
      << number_of_display_filters << endl;
  indent();
  for(unsigned int i = 0; i < number_of_display_filters; ++i)
  {
    //TODO: describeFilter(mData);
  }
  dedent();

  unsigned int number_of_scene_display_parameters = mData.readUnsignedInt();
  cout << getIndent() << "number_of_scene_display_parameters "
      << number_of_scene_display_parameters << endl;
  indent();
  for(unsigned int i = 0; i < number_of_scene_display_parameters; ++i)
  {
    describeSceneDisplayParameters(mData);
  }
  dedent();

  describeUserData(mData);
  dedent();
}

void describeGraphics(BitByBitData &mData)
{
  bool sameGraphicsAsCurrent = mData.readBit();
  cout << getIndent() << "Same graphics as current graphics? "
      << (sameGraphicsAsCurrent?"yes":"no") << endl;
  if(!sameGraphicsAsCurrent)
  {
    layer_index = mData.readUnsignedInt()-1;
    cout << getIndent() << "layer_index " << layer_index << endl;
    index_of_line_style = mData.readUnsignedInt()-1;
    cout << getIndent() << "index_of_line_style "
        << index_of_line_style << endl;
    unsigned char c1 = mData.readChar();
    unsigned char c2 = mData.readChar();
    behaviour_bit_field = c1 | (static_cast<unsigned short>(c2) << 8);
    cout << getIndent() << "behaviour_bit_field "
        << behaviour_bit_field << endl;
  }
}

void describeContentPRCBaseWithGraphics(BitByBitData &mData, bool efr)
{
  describeContentPRCBase(mData,efr);
  describeGraphics(mData);
}

void describePartDefinition(BitByBitData &mData)
{
  cout << getIndent() << "--Part Definition--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_ASM_PartDefinition))
    return;
  indent();

  describeContentPRCBaseWithGraphics(mData,true);
  describeExtent3d(mData);

  unsigned int number_of_representation_items = mData.readUnsignedInt();
  cout << getIndent() << "number_of_representation_items "
      << number_of_representation_items << endl;
  indent();
  for(unsigned int i = 0; i < number_of_representation_items; ++i)
  {
    describeRepresentationItem(mData);
  }
  dedent();

  describeMarkups(mData);

  unsigned int number_of_views = mData.readUnsignedInt();
  cout << getIndent() << "number_of_views " << number_of_views << endl;
  indent();
  for(unsigned int i = 0; i < number_of_views; ++i)
  {
    describeAnnotationView(mData);
  }
  dedent();

  describeUserData(mData);
  dedent();
}

void describeMarkups(BitByBitData& mData)
{
  cout << getIndent() << "--Markups--" << endl;
  indent();

  unsigned int number_of_linked_items = mData.readUnsignedInt();
  cout << getIndent() << "number_of_linked_items "
      << number_of_linked_items << endl;
  for(unsigned int i = 0; i < number_of_linked_items; ++i)
  {
    cout << "describe linked item!" << endl;
  }

  unsigned int number_of_leaders = mData.readUnsignedInt();
  cout << getIndent() << "number_of_leaders " << number_of_leaders << endl;
  for(unsigned int i = 0; i < number_of_leaders; ++i)
  {
    cout << "describe leader!" << endl;
  }

  unsigned int number_of_markups = mData.readUnsignedInt();
  cout << getIndent() << "number_of_markups " << number_of_markups << endl;
  for(unsigned int i=0; i < number_of_markups; ++i)
  {
    cout << "describe markup!" << endl;
  }

  unsigned int number_of_annotation_entities = mData.readUnsignedInt();
  cout << getIndent() << "number_of_annotation_entities "
      << number_of_annotation_entities << endl;
  for(unsigned int i=0; i < number_of_annotation_entities; ++i)
  {
    cout << "describe annotation entity!" << endl;
  }

  dedent();
}

void describeAnnotationView(BitByBitData &mData)
{
  cout << getIndent() << "--Annotation View--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_MKP_View))
    return;
  indent();
  describeContentPRCBaseWithGraphics(mData,true);
  unsigned int number_of_annotations = mData.readUnsignedInt();
  for(unsigned int i = 0; i < number_of_annotations; ++i)
  {
    //TODO: describeReferenceUniqueIdentifier(mData);
  }
  //TODO: describePlane(mData);
  bool scene_display_parameters = mData.readBit();
  if(scene_display_parameters)
  {
    describeSceneDisplayParameters(mData);
  }
  describeUserData(mData);
  dedent();
}

void describeExtent3d(BitByBitData &mData)
{ // I suspect the order of min/max should be flipped
  cout << getIndent() << "Minimum" << endl;
  indent(); describeVector3d(mData); dedent();
  cout << getIndent() << "Maximum" << endl;
  indent(); describeVector3d(mData); dedent();
}

void describeExtent1d(BitByBitData &mData)
{
  cout << getIndent() << "Minimum " << mData.readDouble() << endl;
  cout << getIndent() << "Maximum " << mData.readDouble() << endl;
}

void describeExtent2d(BitByBitData &mData)
{
  cout << getIndent() << "Minimum" << endl;
  indent(); describeVector2d(mData); dedent();
  cout << getIndent() << "Maximum" << endl;
  indent(); describeVector2d(mData); dedent();
}

void describeVector3d(BitByBitData &mData)
{
  double x = mData.readDouble();
  double y = mData.readDouble();
  double z = mData.readDouble();
  cout << getIndent() << '(' << x << ',' << y << ',' << z << ')' << endl;
}

void describeVector2d(BitByBitData &mData)
{
  double x = mData.readDouble();
  double y = mData.readDouble();
  cout << getIndent() << '(' << x << ',' << y << ')' << endl;
}

void describePicture(BitByBitData &mData)
{
  cout << getIndent() << "--Picture--" << endl;
  unsigned int sectionCode = mData.readUnsignedInt();
  if(sectionCode != PRC_TYPE_GRAPH_Picture)
  {
    cout << getIndent() << "Invalid section code." << endl;
  }

  describeContentPRCBase(mData,false);

  int format = mData.readInt();
  switch(format)
  {
    case KEPRCPicture_PNG:
      cout << getIndent() << "PNG format" << endl;
      break;
    case KEPRCPicture_JPG:
      cout << getIndent() << "JPG format" << endl;
      break;
    case KEPRCPicture_BITMAP_RGB_BYTE:
      cout << getIndent() << "gzipped pixel data (see PRC base compression). Each element is a RGB triple. (3 components)" << endl;
      break;
    case KEPRCPicture_BITMAP_RGBA_BYTE:
      cout << getIndent() << "gzipped pixel data (see PRC base compression). Each element is a complete RGBA element. (4 components)" << endl;
      break;
    case KEPRCPicture_BITMAP_GREY_BYTE:
      cout << getIndent() << "gzipped pixel data (see PRC base compression). Each element is a single luminance value. (1 components)" << endl;
      break;
    case KEPRCPicture_BITMAP_GREYA_BYTE:
      cout << getIndent() << "gzipped pixel data (see PRC base compression). Each element is a luminance/alpha pair. (2 components)" << endl;
      break;
    default:
      cout << getIndent() << "Invalid picture format." << endl;
      break;
  }
  cout << getIndent() << "uncompressed_file_index "
      << mData.readUnsignedInt()-1 << endl;
  cout << getIndent() << "pixel width " << mData.readUnsignedInt() << endl;
  cout << getIndent() << "pixel height " << mData.readUnsignedInt() << endl;
}

void describeTextureDefinition(BitByBitData &mData)
{
  cout << getIndent() << "--Texture Definition--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_GRAPH_TextureDefinition))
    return;
  
  cout << getIndent() << "TODO: Can't describe textures yet." << endl;
}

void describeMaterial(BitByBitData &mData)
{
  cout << getIndent() << "--Material--" << endl;
  unsigned int code = mData.readUnsignedInt();
  if(code == PRC_TYPE_GRAPH_Material)
  {
    describeContentPRCBase(mData,true);
    cout << getIndent() << "index of ambient color "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "index of diffuse color "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "index of emissive color "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "index of specular color "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "shininess " << mData.readDouble() << endl;
    cout << getIndent() << "ambient_alpha " << mData.readDouble() << endl;
    cout << getIndent() << "diffuse_alpha " << mData.readDouble() << endl;
    cout << getIndent() << "emissive_alpha " << mData.readDouble() << endl;
    cout << getIndent() << "specular_alpha " << mData.readDouble() << endl;
  }
  else if(code == PRC_TYPE_GRAPH_TextureApplication)
  {
    describeContentPRCBase(mData,true);
    cout << getIndent() << "material_generic_index "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "texture_definition_index "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "next_texture_index "
        << mData.readUnsignedInt() - 1 << endl;
    cout << getIndent() << "UV_coordinates_index "
        << mData.readUnsignedInt() - 1 << endl;
  }
  else
  {
    cout << getIndent() << "Invalid section code in material definition."
        << endl;
  }
}

void describeLinePattern(BitByBitData &mData)
{
  cout << getIndent() << "--Line Pattern--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_GRAPH_LinePattern))
    return;
  indent();

  describeContentPRCBase(mData,true);
  unsigned int size_lengths = mData.readUnsignedInt();
  cout << getIndent() << "size_lengths " << size_lengths << endl;
  indent();
  for(unsigned int i=0;i<size_lengths;i++)
  {
    cout << getIndent() << "length " << mData.readDouble() << endl;
  }
  dedent();
  cout << getIndent() << "phase " << mData.readDouble() << endl;
  cout << getIndent() << "is real length "
      << (mData.readBit()?"yes":"no") << endl;

  dedent();
}

void describeCategory1LineStyle(BitByBitData &mData)
{
  cout << getIndent() << "--Category 1 Line Style--" << endl;
  if(!checkSectionCode(mData,PRC_TYPE_GRAPH_Style))
    return;
  indent();

  // This was missing from the docs!!!!! argh!
  describeContentPRCBase(mData,true); 


  cout << getIndent() << "line_width " << mData.readDouble() << " mm" << endl;

  cout << getIndent() << "is_vpicture " << (mData.readBit()?"yes":"no") << endl;

  cout << getIndent() << "line_pattern_index/vpicture_index "
      << static_cast<int>(mData.readUnsignedInt()-1) << endl;
  cout << getIndent() << "is_material " << (mData.readBit()?"yes":"no") << endl;
  cout << getIndent() << "color_index / material_index "
      << static_cast<int>(mData.readUnsignedInt()-1) << endl;

  bool is_transparency_defined = mData.readBit();
  cout << getIndent() << "is_transparency_defined "
      << (is_transparency_defined?"yes":"no") << endl;
  if(is_transparency_defined)
  {
    indent();
    cout << getIndent() << "transparency "
        << static_cast<unsigned int>(mData.readChar()) << endl;
    dedent();
  }

  bool is_additional_1_defined = mData.readBit();
  cout << getIndent() << "is_additional_1_defined "
      << (is_additional_1_defined?"yes":"no") << endl;
  if(is_additional_1_defined)
  {
    indent();
    cout << getIndent() << "additional_1 "
        << static_cast<unsigned int>(mData.readChar()) << endl;
    dedent();
  }

  bool is_additional_2_defined = mData.readBit();
  cout << getIndent() << "is_additional_2_defined "
      << (is_additional_2_defined?"yes":"no") << endl;
  if(is_additional_2_defined)
  {
    indent();
    cout << getIndent() << "additional_2 "
        << static_cast<unsigned int>(mData.readChar()) << endl;
    dedent();
  }

  bool is_additional_3_defined = mData.readBit();
  cout << getIndent() << "is_additional_3_defined "
      << (is_additional_3_defined?"yes":"no") << endl;
  if(is_additional_3_defined)
  {
    indent();
    cout << getIndent() << "additional_3 "
        << static_cast<unsigned int>(mData.readChar()) << endl;
    dedent();
  }
  dedent();
}

void describeFillPattern(BitByBitData &mData)
{
  cout << getIndent() << "--Fill Pattern--" << endl;
  unsigned int type = mData.readUnsignedInt();
  cout << getIndent() << "type " << type << endl;
  switch(type)
  {
    //TODO: actually describe fill patterns
    default:
      cout << getIndent() << "Invalid fill pattern type " << type << endl;
  }
}

void describeRepresentationItemContent(BitByBitData &mData)
{
  describeContentPRCBaseWithGraphics(mData,true);
  unsigned int index_local_coordinate_system = mData.readUnsignedInt()-1;
  unsigned int index_tessellation = mData.readUnsignedInt()-1;
  //cast to int will not be right for big indices
  cout << getIndent() << "index_local_coordinate_system "
      << static_cast<int>(index_local_coordinate_system) << endl;
  cout << getIndent() << "index_tessellation "
      << static_cast<int>(index_tessellation) << endl;
}

void describeRepresentationItem(BitByBitData &mData)
{
  cout << getIndent() << "--Representation Item--" << endl;
  unsigned int type = mData.readUnsignedInt();
  switch(type)
  {
    case PRC_TYPE_RI_Curve:
    {
      cout << getIndent() << "--PRC_TYPE_RI_Curve--" << endl;
      describeRepresentationItemContent(mData);
      bool has_wire_body = mData.readBit();
      if(has_wire_body)
      {
        cout << getIndent() << "context_id " << mData.readUnsignedInt() << endl;
        cout << getIndent() << "body_id " << mData.readUnsignedInt() << endl;
      }
      describeUserData(mData);
      break;
    }
    case PRC_TYPE_RI_PolyBrepModel:
    {
      cout << getIndent() << "--PRC_TYPE_RI_PolyBrepModel--" << endl;
      describeRepresentationItemContent(mData);
      cout << getIndent() << "is_closed "
          << (mData.readBit()?"yes":"no") << endl;
      describeUserData(mData);
      break;
    }
    case PRC_TYPE_RI_BrepModel:
    {
      cout << getIndent() << "--PRC_TYPE_RI_BrepModel--" << endl;
      describeRepresentationItemContent(mData);
      bool has_brep_data = mData.readBit();
      cout << getIndent() << "has_brep_data "
          << (has_brep_data?"yes":"no") << endl;
      if(has_brep_data)
      {
        cout << getIndent() << "context_id " << mData.readUnsignedInt() << endl;
        cout << getIndent() << "object_id " << mData.readUnsignedInt() << endl;
      }
      cout << getIndent() << "is_closed "
          << (mData.readBit()?"yes":"no") << endl;
      describeUserData(mData);
      break;
    }
    case PRC_TYPE_RI_Direction:

    case PRC_TYPE_RI_Plane:
    case PRC_TYPE_RI_CoordinateSystem:
    case PRC_TYPE_RI_PointSet:
    case PRC_TYPE_RI_Set:
    case PRC_TYPE_RI_PolyWire:
      cout << getIndent() << "TODO: Unhandled representation item "
          << type << endl;
      break;
    default:
      cout << getIndent() << "Invalid representation item type "
          << type << endl;
      break;
  }
}

void describeRGBColour(BitByBitData &mData)
{
  cout << getIndent() << "R: " << mData.readDouble();
  cout << " G: " << mData.readDouble();
  cout << " B: " << mData.readDouble() << endl;
}

void describeSchema(BitByBitData &mData)
{
  cout << getIndent() << "--Schema--" << endl;
  indent();
  unsigned int numSchemas = mData.readUnsignedInt();
  cout << getIndent() << "Number of Schemas " << numSchemas << endl;
  if(numSchemas != 0)
  {
    cout << "Error: Don't know how to handle multiple schemas." << endl;
  }
  dedent();
}

string currentName;
int layer_index;
int index_of_line_style;
unsigned short behaviour_bit_field;

void resetCurrentGraphics()
{
  layer_index = -1;
  index_of_line_style = -1;
  behaviour_bit_field = 1;
}

void unFlushSerialization()
{
  currentName = "";

  resetCurrentGraphics();
}

void describeName(BitByBitData &mData)
{
  bool sameNameAsCurrent = mData.readBit();
  cout << getIndent() << "Same name as current name? "
      << (sameNameAsCurrent?"yes":"no") << endl;
  if(!sameNameAsCurrent)
    currentName = mData.readString();
  cout << getIndent() << "Name \"" << currentName << '\"' << endl;
}

void describeUnit(BitByBitData &mData)
{
  cout << getIndent() << "Unit is from CAD file? "
      << (mData.readBit()?"yes":"no") << endl;
  cout << getIndent() << "Unit is " << mData.readDouble() << " mm" << endl;
}

void describeAttributes(BitByBitData &mData)
{
  cout << getIndent() << "--Attributes--" << endl;
  indent();

  unsigned int numAttribs = mData.readUnsignedInt();
  cout << getIndent() << "Number of Attributes " << numAttribs << endl;
  indent();
  for(unsigned int i = 0; i < numAttribs; ++i)
  {
    cout << getIndent() << "PRC_TYPE_MISC_Attribute "
        << mData.readUnsignedInt() << endl;
    bool titleIsInt = mData.readBit();
    cout << getIndent() << "Title is integer? "
        << (titleIsInt?"yes":"no") << endl;
    indent();
    if(titleIsInt)
    {
      cout << getIndent() << "Title " << mData.readUnsignedInt() << endl;
    }
    else
    {
      cout << getIndent() << "Title \"" << mData.readString() << '\"' << endl;
    }
    unsigned int sizeOfAttributeKeys = mData.readUnsignedInt();
    cout << getIndent() << "Size of Attribute Keys "
        << sizeOfAttributeKeys << endl;
    for(unsigned int a = 0; a < sizeOfAttributeKeys; ++a)
    {
      bool titleIsInt = mData.readBit();
      cout << getIndent() << "Title is integer? "
          << (titleIsInt?"yes":"no") << endl;
      indent();
      if(titleIsInt)
      {
        cout << getIndent() << "Title " << mData.readUnsignedInt() << endl;
      }
      else
      {
        cout << getIndent() << "Title \"" << mData.readString() << '\"' << endl;
      }
      dedent();
      unsigned int attributeType = mData.readUnsignedInt();
      cout << getIndent() << "Attribute Type " << attributeType << endl;
      switch(attributeType)
      {
        case KEPRCModellerAttributeTypeInt:
          cout << getIndent() << "Attribute Value (int) "
              << mData.readInt() << endl;
          break;
        case KEPRCModellerAttributeTypeReal:
          cout << getIndent() << "Attribute Value (double) "
              << mData.readDouble() << endl;
          break;
        case KEPRCModellerAttributeTypeTime:
          cout << getIndent() << "Attribute Value (time_t) "
              << mData.readUnsignedInt() << endl;
          break;
        case KEPRCModellerAttributeTypeString:
          cout << getIndent() << "Attribute Value (string) \""
              << mData.readString() << '\"' << endl;
          break;
        default:
          break;
      }
    }
    dedent();

    cout << endl;
  }
  dedent();


  dedent();
}

void describeContentPRCBase(BitByBitData &mData, bool typeEligibleForReference)
{
  cout << getIndent() << "--ContentPRCBase--" << endl;
  indent();
  describeAttributes(mData);
  describeName(mData);
  if(typeEligibleForReference)
  {
    cout << getIndent() << "CAD_identifier "
        << mData.readUnsignedInt() << endl;
    cout << getIndent() << "CAD_persistent_identifier "
        << mData.readUnsignedInt() << endl;
    cout << getIndent() << "PRC_unique_identifier "
        << mData.readUnsignedInt() << endl;
  }
  dedent();
}

void describeCompressedUniqueID(BitByBitData &mData)
{
  cout << getIndent() << "UUID: " << hex << setfill('0');
  for(int i = 0; i < 4; ++i)
    cout << setw(8) << mData.readUnsignedInt() << ' ';
  cout << dec << setfill(' ') << endl;
}

void describeUserData(BitByBitData &mData)
{
  unsigned int bits = mData.readUnsignedInt();
  cout << getIndent() << bits << " bits of user data" << endl;
  indent();
  for(unsigned int i = 0; i < bits; ++i)
  {
    if(i%64 == 0)
      cout << getIndent();
    cout << mData.readBit();
    if(i%64 == 63)
      cout << endl;
  }
  if(bits%64 != 0)
    cout << endl;
  dedent();
}

bool checkSectionCode(BitByBitData &mData, unsigned int code)
{
  unsigned int num = mData.readUnsignedInt();
  if(code != num)
  {
    cout << getIndent() << "Invalid section code " << num <<
        ". Expected " << code << " at "; mData.tellPosition();
    return false;
  }
  else
  {
    cout << getIndent() << "Section code " << code << endl;
    return true;
  }
}

unsigned int currentIndent = 0;

string getIndent()
{
  ostringstream out;
  for(unsigned int i = 0; i < currentIndent; ++i)
    out << "  ";
  return out.str();
}

void indent()
{
  ++currentIndent;
}

void dedent()
{
  --currentIndent;
}

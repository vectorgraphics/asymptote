/************
*
*   This file is part of a tool for producing 3D content in the PRC format.
*   Copyright (C) 2008  Orest Shardt <shardtor (at) gmail dot com>
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*************/

#include "oPRCFile.h"
#include <time.h>

using std::string;

void PRCline::writeRepresentationItem(PRCbitStream &out,uint32_t index)
{
  out << (uint32_t)(PRC_TYPE_RI_Curve);
  ContentPRCBase(&EMPTY_ATTRIBUTES,"line",true,makeCADID(),0,makePRCID()).write(out);
  writeGraphics(out,-1,parent->getColourIndex(colour),1);
  out << (uint32_t)0 // index_local_coordinate_system+1
      << (uint32_t)0; // index_tessellation

  out << true // has wire body
      << (uint32_t)index+1 // index of context in geometry section
      << (uint32_t)1; // body index in topological context
  UserData(0,0).write(out);
}

void PRCline::writeTopologicalContext(PRCbitStream &out)
{
  // topological context
  out << (uint32_t)(PRC_TYPE_TOPO_Context);
  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint8_t)0 // behaviour
      << 0.0 // granularity
      << 0.001 // tolerance
      << true // have smallest face thickness
      << 0.01 // smallest face thickness
      << false; // have scale

  out << (uint32_t)1; // number of bodies
  // body
  out << (uint32_t)PRC_TYPE_TOPO_SingleWireBody;

  out << false // base topology: no base information
      << (uint8_t)0; // content body: behaviour
  // Not sure what a 0 means, since it satisfies no #define,
  // but I think it means no bbox

  // wire edge
  out << false // not already stored
      << (uint32_t)PRC_TYPE_TOPO_WireEdge
      << false; // base topology: no base information

  // polyline
  out << false // not already stored
      << (uint32_t)PRC_TYPE_CRV_PolyLine
      << false // base topology: no base information
      << KEPRCExtendTypeNone // extend info
      << true // is 3d
      << false // no transformation
      << 0.0 << static_cast<double>(numberOfPoints) // parameterization interval
      << 1.0 // no reparameterization
      << 0.0 // no reparameterization
      << (uint32_t) numberOfPoints;

  // points
  for(uint32_t i = 0; i < numberOfPoints; ++i)
    out << points[i][0] << points[i][1] << points[i][2];

  // ending of wire edge
  out << false; // trim surface domain
}

void PRCline::writeExtraGeometryContext(PRCbitStream &out)
{
  //geometry summary
  out << (uint32_t)1; // number of bodies
  out << (uint32_t)PRC_TYPE_TOPO_SingleWireBody; // body's serial type
  //context graphics
  out << (uint32_t)0; // number of treat types
}

void PRCcurve::writeRepresentationItem(PRCbitStream &out,uint32_t index)
{
  out << (uint32_t)(PRC_TYPE_RI_Curve);
  ContentPRCBase(&EMPTY_ATTRIBUTES,"curve",true,makeCADID(),0,makePRCID()).write(out);
  writeGraphics(out,-1,parent->getColourIndex(colour),1);
  out << (uint32_t)0 // index_local_coordinate_system+1
      << (uint32_t)0; // index_tessellation

  out << true // has wire body
      << (uint32_t)index+1 // index of context in geometry section
      << (uint32_t)1; // body index in topological context
  UserData(0,0).write(out);
}

void PRCcurve::writeTopologicalContext(PRCbitStream &out)
{
  // topological context
  out << (uint32_t)(PRC_TYPE_TOPO_Context);
  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint8_t)0 // behaviour
      << 0.0 // granularity
      << 0.001 // tolerance
      << true // have smallest face thickness
      << 0.01 // smallest face thickness
      << false; // have scale

  out << (uint32_t)1; // number of bodies
  // body
  out << (uint32_t)PRC_TYPE_TOPO_SingleWireBody;

  out << false // base topology: no base information
      << (uint8_t)0; // content body: behaviour
  // Not sure what a 0 means, since it satisfies no #define,
  // but I think it means no bbox

  // wire edge
  out << false // not already stored
      << (uint32_t)PRC_TYPE_TOPO_WireEdge
      << false; // base topology: no base information

  // curve
  out << false // not already stored
      << (uint32_t)PRC_TYPE_CRV_NURBS
      << false // base topology: no base information
      << KEPRCExtendTypeNone // extend info
      << true // is 3D
      << isRational
      << (uint32_t)degree // degree
      << (uint32_t)numberOfControlPoints-1 // control points - 1
      << (uint32_t)degree+numberOfControlPoints; // knots - 1

  // control points
  for(uint32_t i = 0; i < numberOfControlPoints; ++i)
  {
    out << controlPoints[i][0] << controlPoints[i][1] << controlPoints[i][2];
    if(isRational)
      out << weights[i];
  }

  writeKnots(out);

  out << (uint32_t)KEPRCKnotTypeUnspecified // KEPRCKnotTypePiecewiseBezierKnots
      << (uint32_t)KEPRCBSplineCurveFormUnspecified; // curve form

  // ending of wire edge
  out << false; // trim surface domain

}

void PRCcurve::writeExtraGeometryContext(PRCbitStream &out)
{
  //geometry summary
  out << (uint32_t)1; // number of bodies
  out << (uint32_t)PRC_TYPE_TOPO_SingleWireBody; // body's serial type
  //context graphics
  out << (uint32_t)0; // number of treat types
}

void PRCsurface::writeRepresentationItem(PRCbitStream &out,uint32_t index)
{
  out << (uint32_t)(PRC_TYPE_RI_BrepModel);
  ContentPRCBase(&EMPTY_ATTRIBUTES,"surface",true,makeCADID(),0,makePRCID()).write(out);
  writeGraphics(out,0,parent->getColourIndex(colour),1);
  out << (uint32_t)0 // index_local_coordinate_system+1
      << (uint32_t)0; // index_tessellation

  out << true // has brep data
      << (uint32_t)index+1 // index of context in geometry section
      << (uint32_t)1 // body index in topological context
      << false; // is closed???? when is it closed?
  UserData(0,0).write(out);
}

void PRCsurface::writeTopologicalContext(PRCbitStream &out)
{
  // topological context
  out << (uint32_t)(PRC_TYPE_TOPO_Context);
  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint8_t)0 // behaviour
      << 0.0 // granularity
      << 0.001 // tolerance
      << true // have smallest face thickness
      << 0.01 // smallest face thickness
      << false; // have scale

  out << (uint32_t)1; // number of bodies
  // body
  out << (uint32_t)PRC_TYPE_TOPO_BrepData;

  out << false // base topology: no base information
      << (uint8_t)0 // content body: behaviour. Not sure what a 0 means, since it satisfies no #define, but I think it means no bbox
      << (uint32_t)1; // number of connex

  // connex
  out << false // not already stored
      << (uint32_t)PRC_TYPE_TOPO_Connex
      << false // base topology: no base information
      << (uint32_t)1; // number of shells

  // shell
  out << false // not already stored
      << (uint32_t)PRC_TYPE_TOPO_Shell
      << false // base topology: no base information
      << false // shell is closed
      << (uint32_t)1; // number of faces

  // face
  out << false // not already stored
      << (uint32_t)PRC_TYPE_TOPO_Face
      << false; // base topology: no base information

  // NURBS
  out << false // not already stored
      << (uint32_t)PRC_TYPE_SURF_NURBS
      << false // base topology: no base information
      << KEPRCExtendTypeNone // Extend Info
      << isRational // is rational
      << (uint32_t)degreeU // degree in u
      << (uint32_t)degreeV // degree in v
      << (uint32_t)numberOfControlPointsU-1 // control points in u - 1
      << (uint32_t)numberOfControlPointsV-1 // control points in v - 1
      << (uint32_t)degreeU+numberOfControlPointsU // knots in u - 1
      << (uint32_t)degreeV+numberOfControlPointsV; // knots in v -1

  // control points
  for(uint32_t i = 0; i < numberOfControlPointsU*numberOfControlPointsV; ++i)
  {
    out << controlPoints[i][0] << controlPoints[i][1] << controlPoints[i][2];
    if(isRational)
      out << weights[i];
  }

  writeKnots(out);

  out << (uint32_t)KEPRCKnotTypeUnspecified // KEPRCKnotTypePiecewiseBezierKnots
      << (uint32_t)KEPRCBSplineSurfaceFormUnspecified; // surface form

  // ending of face
  out << false // trim surface domain
      << false // have tolerance
      << (uint32_t)0 // number of loops
      << (int32_t)-1; // outer loop index

  // ending of shell
  out << (uint8_t)1;
  // orientation of surface normal w.r.t. face normal:
  // 0: same, 1: opposite, 2: unknown, 1 used in example
}

void PRCsurface::writeExtraGeometryContext(PRCbitStream &out)
{
  //geometry summary
  out << (uint32_t)1; // number of bodies
  out << (uint32_t)PRC_TYPE_TOPO_BrepData; // body's serial type
  //context graphics
  out << (uint32_t)0; // number of treat types
}


void PRCCompressedSection::write(std::ostream &out)
{
  if(prepared)
    out.write((char*)data,getSize());
}

void PRCCompressedSection::prepare()
{
  writeData();
  compress();
  prepared = true;
}

uint32_t PRCCompressedSection::getSize()
{
  if(!prepared)
    return -1;
  else
    return out.getSize();
}

void PRCGlobalsSection::writeData()
{
  // even though this is technically not part of this section,
  // it is handled here for convenience
  out << (uint32_t)(0); // number of schemas
  out << (uint32_t)PRC_TYPE_ASM_FileStructureGlobals;

  SingleAttributeData value = {7094};
  SingleAttribute sa(false,EMPTY_ATTRIBUTE_TITLE,KEPRCModellerAttributeTypeInt,value);
  AttributeTitle iv; iv.text = "__PRC_RESERVED_ATTRIBUTE_PRCInternalVersion";
  Attribute a(false,iv,1,&sa);
  Attributes as(1,&a);
  ContentPRCBase(&as).write(out);
  out << numberOfReferencedFileStructures; // no referencing of file structures
  out << tessellationChordHeightRatio;
  out << tessellationAngleDegrees;
  out << defaultFontFamilyName; // markup serialization helper

  out << numberOfFonts
      << (uint32_t)parent->colourMap.size();
  for(std::vector<RGBAColour>::iterator i = parent->colourMap.begin(); i != parent->colourMap.end(); i++)
  {
    out << i->R << i->G << i->B;
  }
  out << numberOfPictures << numberOfTextureDefinitions << numberOfMaterials;
  out << (uint32_t)1 // number of line patterns hard coded for now
      << (uint32_t)PRC_TYPE_GRAPH_LinePattern;
  ContentPRCBase(&EMPTY_ATTRIBUTES,"",true,makeCADID(),0,makePRCID()).write(out);
  out << (uint32_t)2 // number of lengths
      << 1e6  // size 0
      << 0.0 // size 1
      << 0.0 // phase
      << false; // is real length

  out << (uint32_t) parent->colourMap.size(); // number of styles
  uint32_t index = 0;
  for(std::vector<RGBAColour>::iterator i = parent->colourMap.begin(); i != parent->colourMap.end(); i++, ++index)
  {
    out << (uint32_t)PRC_TYPE_GRAPH_Style;
    ContentPRCBase(&EMPTY_ATTRIBUTES,"",true,makeCADID(),0,makePRCID()).write(out);
    out << 1.0 // line width in mm
        << false // is vpicture
        << (uint32_t)1 // line pattern index+1
        << false // is material
        << (uint32_t)(3*index+1); // color_index+1: the multiplication by three is based on observed data
    if(i->A < 1.0)
      out << true << (uint8_t)(i->A * 256);
    else
      out << false;
    out << false // additional 1 not defined
        << false // additional 2 not defined
        << false; // additional 3 not defined
  }
  out << numberOfFillPatterns 
      << numberOfReferenceCoordinateSystems;
  userData.write(out);
}

void PRCTreeSection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureTree);

  EMPTY_CONTENTPRCBASE.write(out);

  out << (uint32_t)1; // number of part definitions
      // part definitions
  out << (uint32_t)(PRC_TYPE_ASM_PartDefinition);
  ContentPRCBase(&EMPTY_ATTRIBUTES,"",true,makeCADID(),0,makePRCID()).write(out);
  writeGraphics(out,-1,-1,1,true);
  Extent3d(Point3d(1e20,1e20,1e20),Point3d(-1e20,-1e20,-1e20)).write(out);

  out << (uint32_t)parent->fileEntities.size(); // number of representation items
  for(uint32_t i = 0; i < parent->fileEntities.size(); ++i)
  {
    parent->fileEntities[i]->writeRepresentationItem(out,i);
  }

  writeEmptyMarkups(out);

  out << (uint32_t)0; // no views

  UserData(0,0).write(out);
  out << (uint32_t)1; // number of product occurrences
  // only one product occurrence
  out << (uint32_t)(PRC_TYPE_ASM_ProductOccurence);
  SingleAttribute sas[3];
  SingleAttributeData sad;
  AttributeTitle at;

  at.text = "FilePath";
  sad.text = "file name not specified";
  sas[0] = SingleAttribute(false,at,KEPRCModellerAttributeTypeString,sad);

  at.text = "FileSize";
  sad.integer = 1234;
  sas[1] = SingleAttribute(false,at,KEPRCModellerAttributeTypeInt,sad);

  at.text = "FileModificationTime";
  sad.time = time(NULL);
  sas[2] = SingleAttribute(false,at,KEPRCModellerAttributeTypeInt,sad);

  at.text = "__PRC_RESERVED_ATTRIBUTE_A3DF_ProductInformation";
  Attribute attr(false,at,3,sas);
  Attributes attrs(1,&attr);
  ContentPRCBase(&attrs,"Unknown",true,makeCADID(),0,makePRCID()).write(out);

  writeGraphics(out,-1,-1,1,true);
  out << (uint32_t)1 // index_part+1
      << (uint32_t)0 // index_prototype+1
      << (uint32_t)0 // index_external_data+1
      << (uint32_t)0 // number of son product occurrences
      << (uint8_t)0; // product behaviour
  writeUnit(out,true,10.0);
  out << (uint8_t)0 // product information flags
      << (uint32_t)KEPRCProductLoadStatus_Loaded; // product_load_status
  out << false // has location
      << (uint32_t)0; // number of references
  writeEmptyMarkups(out);
  out << (uint32_t)0 // number_of_views
      << false // has entity filter
      << (uint32_t)0 // number_of_display_filters
      << (uint32_t)0; // number_of_scene_display_parameters
  UserData(0,0).write(out);

  // File Structure Internal Data
  out << (uint32_t)(PRC_TYPE_ASM_FileStructure);
  EMPTY_CONTENTPRCBASE.write(out);
  out << makePRCID(); // next available index
  out << (uint32_t)1; // product occurrence index

  UserData(0,0).write(out);
}

void PRCTessellationSection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureTessellation);

  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint32_t)0; // number of tessellations
  UserData(0,0).write(out); // no user data
}

void PRCGeometrySection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureGeometry); 

  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint32_t)parent->fileEntities.size(); // number of topological contexts
  for(uint32_t i = 0; i < parent->fileEntities.size(); ++i)
  {
    parent->fileEntities[i]->writeTopologicalContext(out);
  }

  UserData(0,0).write(out);
}

void PRCExtraGeometrySection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureExtraGeometry);

  EMPTY_CONTENTPRCBASE.write(out);
  out << (uint32_t)parent->fileEntities.size(); // number of contexts
  for(uint32_t i = 0; i < parent->fileEntities.size(); ++i)
  {
    parent->fileEntities[i]->writeExtraGeometryContext(out);
  }

  UserData(0,0).write(out);
}

void PRCModelFile::writeData()
{
  // even though this is technically not part of this section,
  // it is handled here for convenience
  out << (uint32_t)(0); // number of schemas
  out << (uint32_t)(PRC_TYPE_ASM_ModelFile);

  SingleAttributeData value = {7094};
  SingleAttribute sa(false,EMPTY_ATTRIBUTE_TITLE,KEPRCModellerAttributeTypeInt,value);
  AttributeTitle at; at.text = "__PRC_RESERVED_ATTRIBUTE_PRCInternalVersion";
  Attribute a(false,at,1,&sa);
  Attributes as(1,&a);
  ContentPRCBase(&as,"PRC file").write(out);

  writeUnit(out,true,10); // unit is 10 mm, and happens to come from a CAD file

  out << (uint32_t)1; // 1 product occurrence
  //UUID
  out << parent->fileStructures[0]->header.fileStructureUUID[0]
      << parent->fileStructures[0]->header.fileStructureUUID[1]
      << parent->fileStructures[0]->header.fileStructureUUID[2]
      << parent->fileStructures[0]->header.fileStructureUUID[3];
  // index+1
  out << (uint32_t)1;
  // active
  out << true;
  out << (uint32_t)0; // index in model file

  UserData(0,0).write(out);
}

void makeFileUUID(uint32_t *UUID)
{
  // make a UUID
  static uint32_t count = 0;
  ++count;
  // the minimum requirement on UUIDs is that all must be unique in the file
  UUID[0] = 0x33595341; // some constant
  UUID[1] = time(NULL); // the time
  UUID[2] = count;
  UUID[3] = 0xa5a55a5a; // Something random, not seeded by the time, would be nice. But for now, a constant
  // maybe add something else to make it more unique
  // so multiple files can be combined
  // a hash of some data perhaps?
}

void makeAppUUID(uint32_t *UUID)
{
  UUID[0] = UUID[1] = UUID[2] = UUID[3] = 0;
}


void PRCUncompressedFile::write(std::ostream &out)
{
  out.write((char*)&file_size,sizeof(file_size));
  out.write((char*)data,file_size);
}

uint32_t PRCUncompressedFile::getSize()
{
  return sizeof(file_size)+file_size;
}


void PRCStartHeader::write(std::ostream &out)
{
  out.write("PRC",3);
  out.write((char*)&minimal_version_for_read,sizeof(minimal_version_for_read));
  out.write((char*)&authoring_version,sizeof(authoring_version));
  out.write((char*)fileStructureUUID,sizeof(fileStructureUUID));
  out.write((char*)applicationUUID,sizeof(applicationUUID));
}

uint32_t PRCStartHeader::getSize()
{
  return 3+(2+2*4)*sizeof(uint32_t);
}


void PRCFileStructure::write(std::ostream &out)
{
  header.write(out);
  out.write((char*)&number_of_uncompressed_files,sizeof(number_of_uncompressed_files));
  for(uint32_t i = 0; i < number_of_uncompressed_files; ++i)
  {
    uncompressedFiles[i].write(out);
  }
  globals.write(out);
  tree.write(out);
  tessellations.write(out);
  geometry.write(out);
  extraGeometry.write(out);
}

void PRCFileStructure::prepare()
{
  globals.prepare();
  resetGraphicsAndName();

  tree.prepare();
  resetGraphicsAndName();

  tessellations.prepare();
  resetGraphicsAndName();

  geometry.prepare();
  resetGraphicsAndName();

  extraGeometry.prepare();
  resetGraphicsAndName();
}

uint32_t PRCFileStructure::getSize()
{
  uint32_t size = 0;
  size += header.getSize();
  size += sizeof(uint32_t);
  for(uint32_t i = 0; i < number_of_uncompressed_files; ++i)
    size += uncompressedFiles[i].getSize();
  size += globals.getSize();
  size += tree.getSize();
  size += tessellations.getSize();
  size += geometry.getSize();
  size += extraGeometry.getSize();
  return size;
}


void PRCFileStructureInformation::write(std::ostream &out)
{
  out.write((char*)UUID,sizeof(UUID));
  out.write((char*)&reserved,sizeof(reserved));
  out.write((char*)&number_of_offsets,sizeof(number_of_offsets));
  for(uint32_t i = 0; i < number_of_offsets; ++i)
  {
    out.write((char*)(offsets+i),sizeof(uint32_t));
  }
}

uint32_t PRCFileStructureInformation::getSize()
{
  return (4+2+number_of_offsets)*sizeof(uint32_t);
}

void PRCHeader::write(std::ostream &out)
{
  startHeader.write(out);
  out.write((char*)&number_of_file_structures,sizeof(number_of_file_structures));
  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    fileStructureInformation[i].write(out);
  }
  out.write((char*)&model_file_offset,sizeof(model_file_offset));
  out.write((char*)&file_size,sizeof(file_size));
  out.write((char*)&number_of_uncompressed_files,sizeof(number_of_uncompressed_files));
  for(uint32_t i = 0; i < number_of_uncompressed_files; ++i)
  {
    uncompressedFiles[i].write(out);
  }
}

uint32_t PRCHeader::getSize()
{
  uint32_t size = startHeader.getSize() + sizeof(uint32_t);
  for(uint32_t i = 0; i < number_of_file_structures; ++i)
    size += fileStructureInformation[i].getSize();
  size += 3*sizeof(uint32_t);
  for(uint32_t i = 0; i < number_of_uncompressed_files; ++i)
    size += uncompressedFiles[i].getSize();
  return size;
}

bool oPRCFile::add(PRCentity *p)
{
  fileEntities.push_back(p);
  if(getColourIndex(p->colour) == static_cast<uint32_t>(-1))
  {
    colourMap.push_back(p->colour);
  }
  return true;
}

bool oPRCFile::finish()
{
  // only one file structure is currently used
  // prepare data
  fileStructures[0] = new PRCFileStructure(this,0);
  fileStructures[0]->header.minimal_version_for_read = 7094;
  fileStructures[0]->header.authoring_version = 7094;
  makeFileUUID(fileStructures[0]->header.fileStructureUUID);
  makeAppUUID(fileStructures[0]->header.applicationUUID);
  fileStructures[0]->number_of_uncompressed_files = 0;

  // write each section's bit data
  fileStructures[0]->prepare();
  modelFile.prepare();

  // create the header

  // fill out enough info so that sizes can be computed correctly
  header.number_of_uncompressed_files = 0; 
  header.number_of_file_structures = number_of_file_structures;
  header.fileStructureInformation = new PRCFileStructureInformation[number_of_file_structures];
  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    header.fileStructureInformation[i].UUID[0] = fileStructures[i]->header.fileStructureUUID[0];
    header.fileStructureInformation[i].UUID[1] = fileStructures[i]->header.fileStructureUUID[1];
    header.fileStructureInformation[i].UUID[2] = fileStructures[i]->header.fileStructureUUID[2];
    header.fileStructureInformation[i].UUID[3] = fileStructures[i]->header.fileStructureUUID[3];
    header.fileStructureInformation[i].reserved = 0;
    header.fileStructureInformation[i].number_of_offsets = 6;
    header.fileStructureInformation[i].offsets = new uint32_t[6];
  }

  header.startHeader.minimal_version_for_read = 7094;
  header.startHeader.authoring_version = 7094;
  makeFileUUID(header.startHeader.fileStructureUUID);
  makeAppUUID(header.startHeader.applicationUUID);

  header.file_size = getSize();
  header.model_file_offset = header.file_size - modelFile.getSize();

  uint32_t currentOffset = header.getSize();

  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    header.fileStructureInformation[i].offsets[0] = currentOffset; // header offset
    currentOffset += fileStructures[i]->header.getSize() + sizeof(uint32_t);
    for(uint32_t j = 0; j < fileStructures[i]->number_of_uncompressed_files; ++j)
      currentOffset += fileStructures[i]->uncompressedFiles[j].getSize();
    header.fileStructureInformation[i].offsets[1] = currentOffset; // globals offset
    currentOffset += fileStructures[i]->globals.getSize();
    header.fileStructureInformation[i].offsets[2] = currentOffset; // tree offset
    currentOffset += fileStructures[i]->tree.getSize();
    header.fileStructureInformation[i].offsets[3] = currentOffset; // tessellations offset
    currentOffset += fileStructures[i]->tessellations.getSize();
    header.fileStructureInformation[i].offsets[4] = currentOffset; // geometry offset
    currentOffset += fileStructures[i]->geometry.getSize();
    header.fileStructureInformation[i].offsets[5] = currentOffset; // extra geometry offset
    currentOffset += fileStructures[i]->extraGeometry.getSize();
  }

  // write the data
  header.write(output);

  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    fileStructures[i]->write(output);
  }

  modelFile.write(output);
  output.flush();

  for(uint32_t i = 0; i < number_of_file_structures; ++i)
    delete[] header.fileStructureInformation[i].offsets;
  delete[] header.fileStructureInformation;

  return true;
}

uint32_t oPRCFile::getColourIndex(const RGBAColour &c)
{
  for(uint32_t i = 0; i < colourMap.size(); ++i)
  {
    if(colourMap[i] == c)
      return i;
  }
  return -1;
}

uint32_t oPRCFile::getSize()
{
  uint32_t size = header.getSize();

  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    size += fileStructures[i]->getSize();
  }

  size += modelFile.getSize();
  return size;
}

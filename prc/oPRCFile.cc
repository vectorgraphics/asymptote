/************
*
*   This file is part of a tool for producing 3D content in the PRC format.
*   Copyright (C) 2008  Orest Shardt <shardtor (at) gmail dot com>
*   with enhancements contributed by Michail Vidiassov.
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

#include "oPRCFile.h"
#include <time.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <zlib.h>
#include <string.h>

#define WriteUnsignedInteger( value ) out << (uint32_t)(value);
#define WriteInteger( value ) out << (int32_t)(value);
#define SerializeContentPRCBase write(out);
#define SerializeRgbColor( value ) (value).serializeRgbColor(out);
#define SerializePicture( value ) (value).serializePicture(out);
#define SerializeTextureDefinition( value ) (value)->serializeTextureDefinition(out);
#define SerializeMarkup( value ) (value)->serializeMarkup(out);
#define SerializeAnnotationEntity( value ) (value)->serializeAnnotationEntity(out);
#define SerializeFontKeysSameFont( value ) (value).serializeFontKeysSameFont(out);
#define SerializeMaterial( value ) (value)->serializeMaterial(out);

#define SerializeCategory1LineStyle( value ) (value)->serializeCategory1LineStyle(out);
#define SerializeCoordinateSystem( value ) (value)->serializeCoordinateSystem(out);
#define SerializeRepresentationItem( value ) (value)->serializeRepresentationItem(out);
#define SerializePartDefinition( value ) (value)->serializePartDefinition(out);
#define SerializeProductOccurrence( value ) (value)->serializeProductOccurrence(out);
#define SerializeContextAndBodies( value ) (fileStructure->value)->serializeContextAndBodies(out);
#define SerializeGeometrySummary( value ) (fileStructure->value)->serializeGeometrySummary(out);
#define SerializeContextGraphics( value ) (fileStructure->value)->serializeContextGraphics(out);

using std::string;
using namespace std;

void PRCCompressedSection::write(ostream &out)
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
    return m1;
  else
    return out.getSize();
}

void PRCGlobalsSection::writeData()
{
  // even though this is technically not part of this section,
  // it is handled here for convenience
  out << (uint32_t)(0); // number of schemas
  uint32_t i=0; // universal index for PRC standard compatibility
  out << (uint32_t)PRC_TYPE_ASM_FileStructureGlobals;

  PRCSingleAttribute sa((int32_t)PRCVersion);
  PRCAttribute a("__PRC_RESERVED_ATTRIBUTE_PRCInternalVersion");
  a.addKey(sa);
  ContentPRCBase cb;
  cb.addAttribute(a);
  cb.serializeContentPRCBase(out);
  out << numberOfReferencedFileStructures; // no referencing of file structures
  out << tessellationChordHeightRatio;
  out << tessellationAngleDegrees;
  out << defaultFontFamilyName; // markup serialization helper

  const uint32_t number_of_fonts = font_keys_of_font.size();
  WriteUnsignedInteger (number_of_fonts)
  for (i=0;i<number_of_fonts;i++)
  {
    SerializeFontKeysSameFont (font_keys_of_font[i])
  }

  
  const uint32_t number_of_colors = colors.size();
  WriteUnsignedInteger (number_of_colors)
  for (i=0;i<number_of_colors;i++)
      SerializeRgbColor (colors[i])

  const uint32_t number_of_pictures = pictures.size();
  WriteUnsignedInteger (number_of_pictures)
  for (i=0;i<number_of_pictures;i++)
     SerializePicture (pictures[i])

  const uint32_t number_of_texture_definitions = texture_definitions.size();
  WriteUnsignedInteger (number_of_texture_definitions)
  for (i=0;i<number_of_texture_definitions;i++)
     SerializeTextureDefinition (texture_definitions[i])

  const uint32_t number_of_materials = materials.size();
  WriteUnsignedInteger (number_of_materials)
  for (i=0;i<number_of_materials;i++)
     SerializeMaterial (materials[i])

  out << (uint32_t)1 // number of line patterns hard coded for now
      << (uint32_t)PRC_TYPE_GRAPH_LinePattern;
  ContentPRCBase("",true,makeCADID(),0,makePRCID()).serializeContentPRCBase(out);
  out << (uint32_t)0 // number of lengths
      << 0.0 // phase
      << false; // is real length

  const uint32_t number_of_styles = styles.size();
  WriteUnsignedInteger (number_of_styles)
  for (i=0;i<number_of_styles;i++)
     SerializeCategory1LineStyle (styles[i])

  out << numberOfFillPatterns; 

  const uint32_t number_of_reference_coordinate_systems = reference_coordinate_systems.size();
  WriteUnsignedInteger (number_of_reference_coordinate_systems)
  for (i=0;i<number_of_reference_coordinate_systems;i++)
     SerializeCoordinateSystem (reference_coordinate_systems[i])

  userData.write(out);
}

void PRCTreeSection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureTree);

  EMPTY_CONTENTPRCBASE.serializeContentPRCBase(out);

  uint32_t number_of_part_definitions = part_definitions.size();
  WriteUnsignedInteger (number_of_part_definitions) 
  for (uint32_t i=0;i<number_of_part_definitions;i++)
    SerializePartDefinition (part_definitions[i]) 
	
  uint32_t number_of_product_occurrences = product_occurrences.size();
  WriteUnsignedInteger (number_of_product_occurrences) 
  for (uint32_t i=0;i<number_of_product_occurrences;i++)
  {
    product_occurrences[i]->unit_information.unit_from_CAD_file = true;
    product_occurrences[i]->unit_information.unit = unit;
    SerializeProductOccurrence (product_occurrences[i]) 
  }

  // File Structure Internal Data
  out << (uint32_t)(PRC_TYPE_ASM_FileStructure);
  EMPTY_CONTENTPRCBASE.serializeContentPRCBase(out);
  out << makePRCID(); // next available index
  out << (uint32_t)1; // product occurrence index

  UserData(0,0).write(out);
}

void PRCTessellationSection::writeData()
{
  out << (uint32_t)(PRC_TYPE_ASM_FileStructureTessellation);

  EMPTY_CONTENTPRCBASE.serializeContentPRCBase(out);
  const uint32_t number_of_tessellations = tessellations.size();
  WriteUnsignedInteger (number_of_tessellations) 
  for (uint32_t i=0;i<number_of_tessellations;i++)
    tessellations[i]->serializeBaseTessData(out);
  UserData(0,0).write(out); // no user data
}

void PRCGeometrySection::writeData()
{
   WriteUnsignedInteger (PRC_TYPE_ASM_FileStructureGeometry)

  EMPTY_CONTENTPRCBASE.serializeContentPRCBase(out);
   const uint32_t number_of_contexts = fileStructure->contexts.size();
   WriteUnsignedInteger (number_of_contexts)
   for (uint32_t i=0;i<number_of_contexts;i++)
      SerializeContextAndBodies (contexts[i])

  UserData(0,0).write(out);
}

void PRCExtraGeometrySection::writeData()
{
   WriteUnsignedInteger (PRC_TYPE_ASM_FileStructureExtraGeometry)

  EMPTY_CONTENTPRCBASE.serializeContentPRCBase(out);
   const uint32_t number_of_contexts = fileStructure->contexts.size();
   WriteUnsignedInteger (number_of_contexts)
   for (uint32_t i=0;i<number_of_contexts;i++) 
   {
      SerializeGeometrySummary (contexts[i]) 
      SerializeContextGraphics (contexts[i]) 
   }

  UserData(0,0).write(out);
}

void PRCModelFile::writeData()
{
  // even though this is technically not part of this section,
  // it is handled here for convenience
  out << (uint32_t)(0); // number of schemas
  out << (uint32_t)(PRC_TYPE_ASM_ModelFile);

  PRCSingleAttribute sa((int32_t)PRCVersion);
  PRCAttribute a("__PRC_RESERVED_ATTRIBUTE_PRCInternalVersion");
  a.addKey(sa);
  ContentPRCBase cb("PRC file");
  cb.addAttribute(a);
  cb.serializeContentPRCBase(out);

  writeUnit(out,true,unit); // unit is specified, and happens to come from a CAD file

  out << (uint32_t)1; // 1 product occurrence
  //UUID
  out << parent->fileStructures[0]->header.fileStructureUUID[0]
      << parent->fileStructures[0]->header.fileStructureUUID[1]
      << parent->fileStructures[0]->header.fileStructureUUID[2]
      << parent->fileStructures[0]->header.fileStructureUUID[3];
  // index+1
  out << (uint32_t)parent->fileStructures[0]->tree.product_occurrences.size();
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

void writeUINT32_T(ostream &out, uint32_t data)
{
#ifdef WORDS_BIGENDIAN
  out.write(((char*)&data)+3,1);
  out.write(((char*)&data)+2,1);
  out.write(((char*)&data)+1,1);
  out.write(((char*)&data)+0,1);
#else
  out.write(((char*)&data)+0,1);
  out.write(((char*)&data)+1,1);
  out.write(((char*)&data)+2,1);
  out.write(((char*)&data)+3,1);
#endif
}

void PRCUncompressedFile::write(ostream &out)
{
  if(data!=NULL)
  {
    writeUINT32_T(out,file_size);
    out.write((char*)data,file_size);
  }
}

uint32_t PRCUncompressedFile::getSize()
{
  return sizeof(file_size)+file_size;
}


void PRCStartHeader::write(ostream &out)
{
  out.write("PRC",3);
  writeUINT32_T(out,minimal_version_for_read);
  writeUINT32_T(out,authoring_version);
  writeUINT32_T(out,fileStructureUUID[0]);
  writeUINT32_T(out,fileStructureUUID[1]);
  writeUINT32_T(out,fileStructureUUID[2]);
  writeUINT32_T(out,fileStructureUUID[3]);

  writeUINT32_T(out,applicationUUID[0]);
  writeUINT32_T(out,applicationUUID[1]);
  writeUINT32_T(out,applicationUUID[2]);
  writeUINT32_T(out,applicationUUID[3]);
}

uint32_t PRCStartHeader::getSize()
{
  return 3+(2+2*4)*sizeof(uint32_t);
}


void PRCFileStructure::write(ostream &out)
{
  header.write(out);
  uint32_t number_of_uncompressed_files = uncompressedFiles.size();
  writeUINT32_T(out,number_of_uncompressed_files);
  for(list<PRCUncompressedFile>::iterator i = uncompressedFiles.begin(); i != uncompressedFiles.end(); i++)
    i->write(out);
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
  for(list<PRCUncompressedFile>::iterator i = uncompressedFiles.begin(); i != uncompressedFiles.end(); i++)
    size += i->getSize();
  size += globals.getSize();
  size += tree.getSize();
  size += tessellations.getSize();
  size += geometry.getSize();
  size += extraGeometry.getSize();
  return size;
}


void PRCFileStructureInformation::write(ostream &out)
{
  writeUINT32_T(out,UUID[0]);
  writeUINT32_T(out,UUID[1]);
  writeUINT32_T(out,UUID[2]);
  writeUINT32_T(out,UUID[3]);

  writeUINT32_T(out,reserved);
  writeUINT32_T(out,number_of_offsets);
  for(uint32_t i = 0; i < number_of_offsets; ++i)
  {
    writeUINT32_T(out,offsets[i]);
  }
}

uint32_t PRCFileStructureInformation::getSize()
{
  return (4+2+number_of_offsets)*sizeof(uint32_t);
}

void PRCHeader::write(ostream &out)
{
  startHeader.write(out);
  writeUINT32_T(out,number_of_file_structures);
  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    fileStructureInformation[i].write(out);
  }
  writeUINT32_T(out,model_file_offset);
  writeUINT32_T(out,file_size);
  uint32_t number_of_uncompressed_files = uncompressedFiles.size();
  writeUINT32_T(out,number_of_uncompressed_files);
  for(list<PRCUncompressedFile>::iterator i = uncompressedFiles.begin(); i != uncompressedFiles.end(); i++)
    i->write(out);
}

uint32_t PRCHeader::getSize()
{
  uint32_t size = startHeader.getSize() + sizeof(uint32_t);
  for(uint32_t i = 0; i < number_of_file_structures; ++i)
    size += fileStructureInformation[i].getSize();
  size += 3*sizeof(uint32_t);
  for(list<PRCUncompressedFile>::iterator i = uncompressedFiles.begin(); i != uncompressedFiles.end(); i++)
    size += i->getSize();
  return size;
}

void oPRCFile::doGroup(PRCgroup& group, PRCPartDefinition *parent_part_definition, PRCProductOccurrence *parent_product_occurrence)
{
    const string& name = group.name;

    PRCPartDefinition *part_definition = new PRCPartDefinition;

    if(group.options.tess)
    {
      if(!group.lines.empty())
      {
        bool same_color = true;
        const PRCRgbColor &color = group.lines.front().color;
        for(PRCtesslineList::const_iterator lit=group.lines.begin(); lit!=group.lines.end(); lit++)
          if(color!=lit->color)
          {
            same_color = false;
            break;
          }
        map<PRCVector3d,uint32_t> points;
        PRC3DWireTess *tess = new PRC3DWireTess();
        if(!same_color)
        {
          tess->is_segment_color = true;
          tess->is_rgba = false;
        }
        for(PRCtesslineList::const_iterator lit=group.lines.begin(); lit!=group.lines.end(); lit++)
        {
          tess->wire_indexes.push_back(lit->point.size());
          for(uint32_t i=0; i<lit->point.size(); i++)
          {
            map<PRCVector3d,uint32_t>::iterator pPoint = points.find(lit->point[i]);
            if(pPoint!=points.end())
              tess->wire_indexes.push_back(pPoint->second);
            else
            {
              uint32_t point_index = m1;
              points.insert(make_pair(lit->point[i],(point_index = tess->coordinates.size())));
              tess->wire_indexes.push_back(point_index);
              tess->coordinates.push_back(lit->point[i].x);
              tess->coordinates.push_back(lit->point[i].y);
              tess->coordinates.push_back(lit->point[i].z);
            }
            if(!same_color && i>0)
            {
              tess->rgba_vertices.push_back((uint8_t)(255*lit->color.red));
              tess->rgba_vertices.push_back((uint8_t)(255*lit->color.green));
              tess->rgba_vertices.push_back((uint8_t)(255*lit->color.blue));
            }
          }
        }
        const uint32_t tess_index = add3DWireTess(tess);
        PRCPolyWire *polyWire = new PRCPolyWire();
        polyWire->index_tessellation = tess_index;
        if(same_color)
          polyWire->index_of_line_style = addColour(RGBAColour(color.red,color.green,color.blue));
        part_definition->addPolyWire(polyWire);
      }
//    make rectangles pairs of triangles in a tesselation
      if(!group.rectangles.empty())
      {
        bool same_color = true;
        const uint32_t &style = group.rectangles.front().style;
        for(PRCtessrectangleList::const_iterator rit=group.rectangles.begin(); rit!=group.rectangles.end(); rit++)
          if(style!=rit->style)
          {
            same_color = false;
            break;
          }
        map<PRCVector3d,uint32_t> points;
        PRC3DTess *tess = new PRC3DTess();
        PRCTessFace *tessFace = new PRCTessFace();
        tessFace->used_entities_flag=PRC_FACETESSDATA_Triangle;
        uint32_t triangles = 0;
        for(PRCtessrectangleList::const_iterator rit=group.rectangles.begin(); rit!=group.rectangles.end(); rit++)
        {
          const bool degenerate = (rit->vertices[0]==rit->vertices[1]);
          uint32_t vertex_indices[4];
          for(size_t i = (degenerate?1:0); i < 4; ++i)
          {
            map<PRCVector3d,uint32_t>::iterator pPoint = points.find(rit->vertices[i]);
            if(pPoint!=points.end())
              vertex_indices[i] =  pPoint->second;
            else
            {
              points.insert(make_pair(rit->vertices[i],(vertex_indices[i] = tess->coordinates.size())));
              tess->coordinates.push_back(rit->vertices[i].x);
              tess->coordinates.push_back(rit->vertices[i].y);
              tess->coordinates.push_back(rit->vertices[i].z);
            }
          }
          if(degenerate)
          {
            tess->triangulated_index.push_back(vertex_indices[1]);
            tess->triangulated_index.push_back(vertex_indices[2]);
            tess->triangulated_index.push_back(vertex_indices[3]);
            triangles++;
            if(!same_color)
              tessFace->line_attributes.push_back(rit->style);
          }
          else
          {
            tess->triangulated_index.push_back(vertex_indices[0]);
            tess->triangulated_index.push_back(vertex_indices[2]);
            tess->triangulated_index.push_back(vertex_indices[3]);
            triangles++;
            if(!same_color)
              tessFace->line_attributes.push_back(rit->style);
            tess->triangulated_index.push_back(vertex_indices[3]);
            tess->triangulated_index.push_back(vertex_indices[1]);
            tess->triangulated_index.push_back(vertex_indices[0]);
            triangles++;
            if(!same_color)
              tessFace->line_attributes.push_back(rit->style);
          }
        }
        tessFace->sizes_triangulated.push_back(triangles);
        tess->addTessFace(tessFace);
        const uint32_t tess_index = add3DTess(tess);
        PRCPolyBrepModel *polyBrepModel = new PRCPolyBrepModel();
        polyBrepModel->index_tessellation = tess_index;
        polyBrepModel->is_closed = group.options.closed;
        if(same_color)
          polyBrepModel->index_of_line_style = style;
        part_definition->addPolyBrepModel(polyBrepModel);
      }
    }
   
    if(!group.points.empty())
    {
      for(PRCpointsetMap::const_iterator pit=group.points.begin(); pit!=group.points.end(); pit++)
      {
        PRCPointSet *pointset = new PRCPointSet();
        pointset->index_of_line_style = pit->first;
        pointset->point = pit->second;
        part_definition->addPointSet(pointset);
      }
    }

    if(!group.wires.empty())
    {
      PRCTopoContext *wireContext = NULL;
      uint32_t context_index = getTopoContext(wireContext);
      for(PRCwireList::iterator wit=group.wires.begin(); wit!=group.wires.end(); wit++)
      {
        PRCWireEdge *wireEdge = new PRCWireEdge;
        wireEdge->curve_3d = wit->curve;
        PRCSingleWireBody *wireBody = new PRCSingleWireBody;
        wireBody->setWireEdge(wireEdge);
        const uint32_t wire_body_index = wireContext->addSingleWireBody(wireBody);
        PRCWire *wire = new PRCWire();
        wire->index_of_line_style = wit->style;
        wire->context_id = context_index;
        wire->body_id = wire_body_index;
        if(wit->transform)
            wire->index_local_coordinate_system = addTransform(wit->transform);
        part_definition->addWire(wire);
      }
    }

    PRCfaceList &faces = group.faces;
    if(!faces.empty())
    {
      bool same_color = true;
      uint32_t style = faces.front().style;
      for(PRCfaceList::const_iterator fit=faces.begin(); fit!=faces.end(); fit++)
        if(style!=fit->style)
        {
          same_color = false;
          break;
        }
      PRCTopoContext *context = NULL;
      const uint32_t context_index = getTopoContext(context);
      context->granularity = group.options.granularity;
   // Acrobat 9 also does the following:
   // context->tolerance = group.options.granularity;
   // context->have_smallest_face_thickness = true;
   // context->smallest_thickness = group.options.granularity;
      PRCShell *shell = new PRCShell;

      for(PRCfaceList::iterator fit=faces.begin(); fit!=faces.end(); fit++)
      {
        if(fit->transform || group.options.do_break || 
           (fit->transparent && !group.options.no_break))
        {
          PRCShell *shell = new PRCShell;
          shell->addFace(fit->face);
          PRCConnex *connex = new PRCConnex;
          connex->addShell(shell);
          PRCBrepData *body = new PRCBrepData;
          body->addConnex(connex);
          const uint32_t body_index = context->addBrepData(body);
      
          PRCBrepModel *brepmodel = new PRCBrepModel();
          brepmodel->index_of_line_style = fit->style;
          brepmodel->context_id = context_index;
          brepmodel->body_id = body_index;
          brepmodel->is_closed = group.options.closed;

          brepmodel->index_local_coordinate_system = addTransform(fit->transform);

          part_definition->addBrepModel(brepmodel);
        }
        else
        {
          if(!same_color)
            fit->face->index_of_line_style = fit->style;
          shell->addFace(fit->face);
        }
      }
      if(shell->face.empty())
      {
        delete shell;
      }
      else
      {
        PRCConnex *connex = new PRCConnex;
        connex->addShell(shell);
        PRCBrepData *body = new PRCBrepData;
        body->addConnex(connex);
        const uint32_t body_index = context->addBrepData(body);
        PRCBrepModel *brepmodel = new PRCBrepModel();
        if(same_color)
          brepmodel->index_of_line_style = style;
        brepmodel->context_id = context_index;
        brepmodel->body_id = body_index;
        brepmodel->is_closed = group.options.closed;
        part_definition->addBrepModel(brepmodel);
      }
    }

    PRCcompfaceList &compfaces = group.compfaces;
    if(!compfaces.empty())
    {
      bool same_color = true;
      uint32_t style = compfaces.front().style;
      for(PRCcompfaceList::const_iterator fit=compfaces.begin(); fit!=compfaces.end(); fit++)
        if(style!=fit->style)
        {
          same_color = false;
          break;
        }
      PRCTopoContext *context = NULL;
      const uint32_t context_index = getTopoContext(context);
      PRCCompressedBrepData *body = new PRCCompressedBrepData;
      
      body->serial_tolerance=group.options.compression;
      body->brep_data_compressed_tolerance=0.1*group.options.compression;

      for(PRCcompfaceList::const_iterator fit=compfaces.begin(); fit!=compfaces.end(); fit++)
      {
        if(group.options.do_break || 
           (fit->transparent && !group.options.no_break))
        {
          PRCCompressedBrepData *body = new PRCCompressedBrepData;
          body->face.push_back(fit->face);

          body->serial_tolerance=group.options.compression;
          body->brep_data_compressed_tolerance=2.8346456*
            group.options.compression;
          const uint32_t body_index = context->addCompressedBrepData(body);

          PRCBrepModel *brepmodel = new PRCBrepModel();
          brepmodel->index_of_line_style = fit->style;
          brepmodel->context_id = context_index;
          brepmodel->body_id = body_index;
          brepmodel->is_closed = group.options.closed;

          part_definition->addBrepModel(brepmodel);
        }
        else
        {
          if(!same_color)
            fit->face->index_of_line_style = fit->style;
          body->face.push_back(fit->face);
        }
      }
      if(body->face.empty())
      {
        delete body;
      }
      else
      {
        const uint32_t body_index = context->addCompressedBrepData(body);
        PRCBrepModel *brepmodel = new PRCBrepModel();
        if(same_color)
          brepmodel->index_of_line_style = style;
        brepmodel->context_id = context_index;
        brepmodel->body_id = body_index;
        brepmodel->is_closed = group.options.closed;
        part_definition->addBrepModel(brepmodel);
      }
    }

    PRCProductOccurrence *product_occurrence = new PRCProductOccurrence(name);

    for(PRCgroupList::iterator it=group.groupList.begin(); it!=group.groupList.end(); it++)
    {
       doGroup(*it, part_definition, product_occurrence);
    }

    // Simplify and reduce to as simple entities as possible
    // First option - reduce to one element in parent
    if (parent_part_definition && product_occurrence->index_son_occurrence.empty() &&
        part_definition->representation_item.size() == 1 &&
        ( name.empty() || part_definition->representation_item.front()->name.empty() ) &&
        ( !group.transform  || part_definition->representation_item.front()->index_local_coordinate_system==m1) )
    {
      if(part_definition->representation_item.front()->name.empty() )
        part_definition->representation_item.front()->name = name;
      if(part_definition->representation_item.front()->index_local_coordinate_system==m1)
        part_definition->representation_item.front()->index_local_coordinate_system = addTransform(group.transform);
      parent_part_definition->addRepresentationItem(part_definition->representation_item.front());
      part_definition->representation_item.clear();
      delete product_occurrence;
      delete part_definition;
    }
    // Second option - reduce to a set
    else if (parent_part_definition && product_occurrence->index_son_occurrence.empty() &&
      !part_definition->representation_item.empty() &&
      !group.options.do_break )
    {
      PRCSet *set = new PRCSet(name);
      set->index_local_coordinate_system = addTransform(group.transform);
      for(PRCRepresentationItemList::iterator it=part_definition->representation_item.begin(); it!=part_definition->representation_item.end(); it++)
        set->addRepresentationItem(*it);
      part_definition->representation_item.clear();
      parent_part_definition->addSet(set);
      delete product_occurrence;
      delete part_definition;      
    }
    // Third option - create product
    else if ( !product_occurrence->index_son_occurrence.empty() || !part_definition->representation_item.empty())
    {
      if (part_definition->representation_item.empty())
        delete part_definition;
      else
        product_occurrence->index_part = addPartDefinition(part_definition);
      if (group.transform) {
        product_occurrence->location = group.transform;
        group.transform = NULL;
      }
      if (parent_product_occurrence) {
        parent_product_occurrence->index_son_occurrence.push_back(addProductOccurrence(product_occurrence));
      }
      else {
        addProductOccurrence(product_occurrence);
      }
    }
    // Last case - absolutely nothing to do
    else
    {
      delete product_occurrence;
      delete part_definition;
    }
    
}

bool oPRCFile::finish()
{
  rootGroup.name = "root";
  doGroup(rootGroup, NULL, NULL);

  // write each section's bit data
  fileStructures[0]->prepare();
  modelFile.prepare();

  // create the header

  // fill out enough info so that sizes can be computed correctly
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

  header.startHeader.minimal_version_for_read = PRCVersion;
  header.startHeader.authoring_version = PRCVersion;
  makeFileUUID(header.startHeader.fileStructureUUID);
  makeAppUUID(header.startHeader.applicationUUID);

  header.file_size = getSize();
  header.model_file_offset = header.file_size - modelFile.getSize();

  uint32_t currentOffset = header.getSize();

  for(uint32_t i = 0; i < number_of_file_structures; ++i)
  {
    header.fileStructureInformation[i].offsets[0] = currentOffset; // header offset
    currentOffset += fileStructures[i]->header.getSize() + sizeof(uint32_t);
    for(list<PRCUncompressedFile>::iterator j = fileStructures[i]->uncompressedFiles.begin(); j != fileStructures[i]->uncompressedFiles.end(); j++)
      currentOffset += j->getSize();
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

#define REPORT_ERROR( value ) { cerr << value << endl; return m1; }

uint32_t PRCFileStructure::addPicture(EPRCPictureDataFormat format, uint32_t size, const uint8_t *p, uint32_t width, uint32_t height, string name)
{
  if(size==0 || p==NULL)
    REPORT_ERROR( "image not set" )
  PRCPicture picture(name);
  PRCUncompressedFile uncompressedFile;
  uint32_t components=0;
  uint8_t *data = NULL;
  switch(format)
  {
    case KEPRCPicture_BITMAP_RGB_BYTE:
      components = 3;
    case KEPRCPicture_BITMAP_RGBA_BYTE:
      components = 4;
    case KEPRCPicture_BITMAP_GREY_BYTE:
      components = 1;
    case KEPRCPicture_BITMAP_GREYA_BYTE:
      components = 2;
      if(width==0 || height==0)
        REPORT_ERROR( "width or height parameter not set" )
      if (size < width*height*components)
        REPORT_ERROR( "image too small" )
   
      {
        uint32_t compressedDataSize = 0;
        const int CHUNK= 1024; // is this reasonable?
       
        z_stream strm;
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;
        if(deflateInit(&strm,Z_DEFAULT_COMPRESSION) != Z_OK)
          REPORT_ERROR ( "Compression initialization failed" )
        unsigned int sizeAvailable = deflateBound(&strm,size);
        uint8_t *compressedData = (uint8_t*) malloc(sizeAvailable);
        strm.avail_in = size;
        strm.next_in = (unsigned char*)p;
        strm.next_out = (unsigned char*)compressedData;
        strm.avail_out = sizeAvailable;
       
        int code;
        unsigned int chunks = 0;
        while((code = deflate(&strm,Z_FINISH)) == Z_OK)
        {
          ++chunks;
          // strm.avail_out should be 0 if we got Z_OK
          compressedDataSize = sizeAvailable - strm.avail_out;
          compressedData = (uint8_t*) realloc(compressedData,CHUNK*chunks);
          strm.next_out = (Bytef*)(compressedData + compressedDataSize);
          strm.avail_out += CHUNK;
          sizeAvailable += CHUNK;
        }
        compressedDataSize = sizeAvailable-strm.avail_out;
       
        if(code != Z_STREAM_END)
        {
          deflateEnd(&strm);
          free(compressedData);
          REPORT_ERROR ( "Compression error" )
        }
       
        deflateEnd(&strm);
        size = compressedDataSize;
        data = new uint8_t[compressedDataSize];
        memcpy(data, compressedData, compressedDataSize);
        free(compressedData);
      }
      uncompressedFiles.push_back(uncompressedFile);
      uncompressedFiles.back().file_size = size;
      uncompressedFiles.back().data = data;
      picture.format = format;
      picture.uncompressed_file_index = uncompressedFiles.size()-1;
      picture.pixel_width = width;
      picture.pixel_height = height;
      globals.pictures.push_back(picture);
      return globals.pictures.size()-1;
      break;

    case KEPRCPicture_PNG:
    case KEPRCPicture_JPG:
      data = new uint8_t[size];
      memcpy(data, p, size);
      uncompressedFiles.push_back(uncompressedFile);
      uncompressedFiles.back().file_size = size;
      uncompressedFiles.back().data = data;
      picture.format = format;
      picture.uncompressed_file_index = uncompressedFiles.size()-1;
      picture.pixel_width = 0; // width and height are ignored for JPG and PNG pictures - but let us keep things clean
      picture.pixel_height = 0;
      globals.pictures.push_back(picture);
      return globals.pictures.size()-1;
      break;

    default:
      REPORT_ERROR( "unknown picture format" )
  }
  return m1;
}
#undef REPORT_ERROR

uint32_t PRCFileStructure::addTextureDefinition(PRCTextureDefinition *pTextureDefinition)
{
  globals.texture_definitions.push_back(pTextureDefinition);
  return globals.texture_definitions.size()-1;
}

uint32_t PRCFileStructure::addRgbColor(const PRCRgbColor &color)
{
  globals.colors.push_back(color);
  return 3*(globals.colors.size()-1);
}

uint32_t PRCFileStructure::addRgbColorUnique(const PRCRgbColor &color)
{
  for(uint32_t i = 0; i < globals.colors.size(); ++i)
  {
    if(globals.colors[i] == color)
      return 3*i;
  }
  globals.colors.push_back(color);
  return 3*(globals.colors.size()-1);
}

uint32_t oPRCFile::addColor(const PRCRgbColor &color)
{
  PRCcolorMap::iterator pColor = colorMap.find(color);
  uint32_t color_index = m1;
  if(pColor!=colorMap.end())
    return pColor->second;
//  color_index = addRgbColorUnique(color);
  color_index = fileStructures[0]->addRgbColor(color);
  colorMap.insert(make_pair(color,color_index));
  return color_index;
}

uint32_t oPRCFile::addColour(const RGBAColour &colour)
{
  PRCcolourMap::iterator pColour = colourMap.find(colour);
  if(pColour!=colourMap.end())
    return pColour->second;
  const uint32_t color_index = addColor(PRCRgbColor(colour.R, colour.G, colour.B));
  PRCStyle *style = new PRCStyle();  
  style->line_width = 1.0;
  style->is_vpicture = false;
  style->line_pattern_vpicture_index = 0;
  style->is_material = false;
  style->color_material_index = color_index;
  style->is_transparency_defined = (colour.A < 1.0);
  style->transparency = (uint8_t)(colour.A * 256);
  style->additional = 0;
  const uint32_t style_index = fileStructures[0]->addStyle(style);
  colourMap.insert(make_pair(colour,style_index));
  return style_index;
}

uint32_t oPRCFile::addTransform(PRCGeneralTransformation3d*& transform)
{
  if(!transform)
    return m1;
  PRCtransformMap::iterator pTransform = transformMap.find(*transform);
  if(pTransform!=transformMap.end())
    return pTransform->second;
  PRCCoordinateSystem *coordinateSystem = new PRCCoordinateSystem();
  coordinateSystem->axis_set = transform;
  const uint32_t coordinate_system_index = fileStructures[0]->addCoordinateSystem(coordinateSystem);
  transformMap.insert(make_pair(*transform,coordinate_system_index));
  transform = NULL;
  return coordinate_system_index;
}

uint32_t oPRCFile::addMaterial(const PRCmaterial &material)
{
  PRCmaterialMap::iterator pMaterial = materialMap.find(material);
  if(pMaterial!=materialMap.end())
    return pMaterial->second;
  PRCMaterialGeneric *materialGeneric = new PRCMaterialGeneric(); 
  const PRCRgbColor ambient(material.ambient.R, material.ambient.G, material.ambient.B);
  materialGeneric->ambient = addColor(ambient);
  const PRCRgbColor diffuse(material.diffuse.R, material.diffuse.G, material.diffuse.B);
  materialGeneric->diffuse = addColor(diffuse);
  const PRCRgbColor emissive(material.emissive.R, material.emissive.G, material.emissive.B);
  materialGeneric->emissive = addColor(emissive);
  const PRCRgbColor specular(material.specular.R, material.specular.G, material.specular.B);
  materialGeneric->specular = addColor(specular);
  materialGeneric->shininess = material.shininess;
  materialGeneric->ambient_alpha = material.ambient.A;
  materialGeneric->diffuse_alpha = material.diffuse.A;
  materialGeneric->emissive_alpha = material.emissive.A;
  materialGeneric->specular_alpha = material.specular.A;
  const uint32_t material_index = fileStructures[0]->addMaterialGeneric(materialGeneric);
  PRCStyle *style = new PRCStyle();  
  style->line_width = 0.0;
  style->is_vpicture = false;
  style->line_pattern_vpicture_index = 0;
  style->is_material = true;
  style->color_material_index = material_index;
  style->is_transparency_defined = (material.alpha < 1.0);
  style->transparency = (uint8_t)(material.alpha * 256);
  style->additional = 0;
  const uint32_t style_index = fileStructures[0]->addStyle(style);
  materialMap.insert(make_pair(material,style_index)).first;
  return style_index;
}

bool isid(const double t[][4])
{
  return t == NULL ||
    (t[0][0]==1 && t[0][1]==0 && t[0][2]==0 && t[0][3]==0 &&
     t[1][0]==0 && t[1][1]==1 && t[1][2]==0 && t[1][3]==0 &&
     t[2][0]==0 && t[2][1]==0 && t[2][2]==1 && t[2][3]==0 &&
     t[3][0]==0 && t[3][1]==0 && t[3][2]==0 && t[3][3]==1);
}
bool isid(const double t[])
{
  return(isid((double (*)[4])t));
}


void oPRCFile::begingroup(const char *name, PRCoptions *options,
                          const double t[][4])
{
  if(currentGroups.empty())
    currentGroups.push(rootGroup.groupList.insert(rootGroup.groupList.end(),PRCgroup()));
  else
    currentGroups.push(currentGroups.top()->groupList.insert(currentGroups.top()->groupList.end(),PRCgroup()));
  PRCgroup &group = *(currentGroups.top());
  group.name=name;
  if(options) group.options=*options;
  if(t&&!isid(t))
    group.transform = new PRCGeneralTransformation3d(t);
}

void oPRCFile::endgroup()
{
  if(!currentGroups.empty())
    currentGroups.pop();
}

PRCgroup& oPRCFile::findGroup()
{
  if(!currentGroups.empty())
    return *(currentGroups.top());
  else
    return rootGroup;
}

#define ADDWIRE(curvtype)                                 \
  PRCgroup &group = findGroup();                          \
  group.wires.push_back(PRCwire());                       \
  PRCwire &wire = group.wires.back();                     \
  curvtype *curve = new curvtype;                         \
  wire.curve = curve;                                     \
  wire.style = addColour(c);

#define ADDFACE(surftype)                                 \
  PRCgroup &group = findGroup();                          \
  group.faces.push_back(PRCface());                       \
  PRCface& face = group.faces.back();                     \
  surftype *surface = new surftype;                       \
  face.face = new PRCFace;                                \
  face.face->base_surface = surface;                      \
  face.transparent = m.alpha < 1.0;                       \
  face.style = addMaterial(m);

#define ADDCOMPFACE                                       \
  PRCgroup &group = findGroup();                          \
  group.compfaces.push_back(PRCcompface());               \
  PRCcompface& face = group.compfaces.back();             \
  PRCCompressedFace *compface = new PRCCompressedFace;    \
  face.face = compface;                                   \
  face.transparent = m.alpha < 1.0;                       \
  face.style = addMaterial(m);

void oPRCFile::addPoint(const double P[3], const RGBAColour &c)
{
  PRCgroup &group = findGroup();
  group.points[addColour(c)].push_back(PRCVector3d(P[0],P[1],P[2]));
}

void oPRCFile::addLine(uint32_t n, const double P[][3], const RGBAColour &c)
{
  PRCgroup &group = findGroup();
  if(group.options.tess)
  {
    group.lines.push_back(PRCtessline());
    group.lines.back().color.red   = c.R;
    group.lines.back().color.green = c.G;
    group.lines.back().color.blue  = c.B;
    for(uint32_t i=0; i<n; i++)
     group.lines.back().point.push_back(PRCVector3d(P[i][0],P[i][1],P[i][2]));
  }
  else
  {
    ADDWIRE(PRCPolyLine)
    curve->point.resize(n);
    for(uint32_t i=0; i<n; i++)
     curve->point[i].Set(P[i][0],P[i][1],P[i][2]);
    curve->interval.min = 0;
    curve->interval.max = curve->point.size()-1;
  }
}

void oPRCFile::addBezierCurve(uint32_t n, const double cP[][3],
                              const RGBAColour &c)
{
  ADDWIRE(PRCNURBSCurve)
  curve->is_rational = false;
  curve->degree = 3;
  const size_t NUMBER_OF_POINTS = n;
  curve->control_point.resize(NUMBER_OF_POINTS);
  for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
    curve->control_point[i].Set(cP[i][0],cP[i][1],cP[i][2]);
  curve->knot.resize(3+NUMBER_OF_POINTS+1);
  curve->knot[0] = 1;
  for(size_t i = 1; i < 3+NUMBER_OF_POINTS; ++i)
    curve->knot[i] = (i+2)/3; // integer division is intentional
  curve->knot[3+NUMBER_OF_POINTS] = (3+NUMBER_OF_POINTS+1)/3;
}

void oPRCFile::addCurve(uint32_t d, uint32_t n, const double cP[][3], const double *k, const RGBAColour &c, const double w[])
{
  ADDWIRE(PRCNURBSCurve)
  curve->is_rational = w;
  curve->degree = d;
  curve->control_point.resize(n);
  for(uint32_t i = 0; i < n; i++)
    if(w)
      curve->control_point[i].Set(cP[i][0]*w[i],cP[i][1]*w[i],cP[i][2]*w[i],w[i]);
    else
      curve->control_point[i].Set(cP[i][0],cP[i][1],cP[i][2]);
  curve->knot.resize(d+n+1);
  for(uint32_t i = 0; i < d+n+1; i++)
    curve->knot[i] = k[i];
}

void oPRCFile::addRectangle(const double P[][3], const PRCmaterial &m)
{
  PRCgroup &group = findGroup();
  if(group.options.tess)
  {
    PRCgroup &group = findGroup();
    group.rectangles.push_back(PRCtessrectangle());
    PRCtessrectangle &rectangle = group.rectangles.back();
    rectangle.style = addMaterial(m);
    for(size_t i = 0; i < 4; i++)
    {
       rectangle.vertices[i].x = P[i][0];
       rectangle.vertices[i].y = P[i][1];
       rectangle.vertices[i].z = P[i][2];
    }
  }
  else if(group.options.compression == 0.0)
  {
    ADDFACE(PRCNURBSSurface)

    surface->is_rational = false;
    surface->degree_in_u = 1;
    surface->degree_in_v = 1;
    surface->control_point.resize(4);
    for(size_t i = 0; i < 4; ++i)
    {
        surface->control_point[i].x = P[i][0];
        surface->control_point[i].y = P[i][1];
        surface->control_point[i].z = P[i][2];
    }
    surface->knot_u.resize(4);
    surface->knot_v.resize(4);
    surface->knot_v[0] = surface->knot_u[0] = 1;
    surface->knot_v[1] = surface->knot_u[1] = 3;
    surface->knot_v[2] = surface->knot_u[2] = 4;
    surface->knot_v[3] = surface->knot_u[3] = 4;
  }
  else
  {
    ADDCOMPFACE

    compface->degree = 1;
    compface->control_point.resize(4);
    for(size_t i = 0; i < 4; ++i)
    {
        compface->control_point[i].x = P[i][0];
        compface->control_point[i].y = P[i][1];
        compface->control_point[i].z = P[i][2];
    }
  }
}

void oPRCFile::addPatch(const double cP[][3], const PRCmaterial &m)
{
  PRCgroup &group = findGroup();
  if(group.options.compression == 0.0)
  {
    ADDFACE(PRCNURBSSurface)
   
    surface->is_rational = false;
    surface->degree_in_u = 3;
    surface->degree_in_v = 3;
    surface->control_point.resize(16);
    for(size_t i = 0; i < 16; ++i)
    {
        surface->control_point[i].x = cP[i][0];
        surface->control_point[i].y = cP[i][1];
        surface->control_point[i].z = cP[i][2];
    }
    surface->knot_u.resize(8);
    surface->knot_v.resize(8);
    surface->knot_v[0] = surface->knot_u[0] = 1;
    surface->knot_v[1] = surface->knot_u[1] = 1;
    surface->knot_v[2] = surface->knot_u[2] = 1;
    surface->knot_v[3] = surface->knot_u[3] = 1;
    surface->knot_v[4] = surface->knot_u[4] = 2;
    surface->knot_v[5] = surface->knot_u[5] = 2;
    surface->knot_v[6] = surface->knot_u[6] = 2;
    surface->knot_v[7] = surface->knot_u[7] = 2;
  }
  else
  {
    ADDCOMPFACE

    compface->degree = 3;
    compface->control_point.resize(16);
    for(size_t i = 0; i < 16; ++i)
    {
        compface->control_point[i].x = cP[i][0];
        compface->control_point[i].y = cP[i][1];
        compface->control_point[i].z = cP[i][2];
    }
  }
}

void oPRCFile::addSurface(uint32_t dU, uint32_t dV, uint32_t nU, uint32_t nV,
                          const double cP[][3], const double *kU,
                          const double *kV, const PRCmaterial &m,
                          const double w[])
{
  ADDFACE(PRCNURBSSurface)

  surface->is_rational = w;
  surface->degree_in_u = dU;
  surface->degree_in_v = dV;
  surface->control_point.resize(nU*nV);
  for(size_t i = 0; i < nU*nV; i++)
    if(w)
      surface->control_point[i]=PRCControlPoint(cP[i][0]*w[i],cP[i][1]*w[i],cP[i][2]*w[i],w[i]);
    else
      surface->control_point[i]=PRCControlPoint(cP[i][0],cP[i][1],cP[i][2]);
  surface->knot_u.insert(surface->knot_u.end(), kU, kU+(dU+nU+1));
  surface->knot_v.insert(surface->knot_v.end(), kV, kV+(dV+nV+1));
}

#define SETTRANSF \
  if(t&&!isid(t))                                                             \
    face.transform = new PRCGeneralTransformation3d(t);                       \
  if(origin) surface->origin.Set(origin[0],origin[1],origin[2]);              \
  if(x_axis) surface->x_axis.Set(x_axis[0],x_axis[1],x_axis[2]);              \
  if(y_axis) surface->y_axis.Set(y_axis[0],y_axis[1],y_axis[2]);              \
  surface->scale = scale;                                                     \
  surface->geometry_is_2D = false;                                            \
  if(surface->origin!=PRCVector3d(0,0,0))                                     \
    surface->behaviour = surface->behaviour | PRC_TRANSFORMATION_Translate;   \
  if(surface->x_axis!=PRCVector3d(1,0,0)||surface->y_axis!=PRCVector3d(0,1,0)) \
    surface->behaviour = surface->behaviour | PRC_TRANSFORMATION_Rotate;      \
  if(surface->scale!=1)                                                       \
    surface->behaviour = surface->behaviour | PRC_TRANSFORMATION_Scale;       \
  surface->has_transformation = (surface->behaviour != PRC_TRANSFORMATION_Identity);

#define PRCFACETRANSFORM const double origin[3], const double x_axis[3], const double y_axis[3], double scale, const double t[][4]

void oPRCFile::addTube(uint32_t n, const double cP[][3], const double oP[][3], bool straight, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCBlend01)
  SETTRANSF
  if(straight)
  {
    PRCPolyLine *center_curve = new PRCPolyLine;
    center_curve->point.resize(n);
    for(uint32_t i=0; i<n; i++)
      center_curve->point[i].Set(cP[i][0],cP[i][1],cP[i][2]);
    center_curve->interval.min = 0;
    center_curve->interval.max = center_curve->point.size()-1;
    surface->center_curve = center_curve;

    PRCPolyLine *origin_curve = new PRCPolyLine;
    origin_curve->point.resize(n);
    for(uint32_t i=0; i<n; i++)
      origin_curve->point[i].Set(oP[i][0],oP[i][1],oP[i][2]);
    origin_curve->interval.min = 0;
    origin_curve->interval.max = origin_curve->point.size()-1;
    surface->origin_curve = origin_curve;

    surface->uv_domain.min.x = 0;
    surface->uv_domain.max.x = 2*pi;
    surface->uv_domain.min.y = 0;
    surface->uv_domain.max.y = n-1;
  }
  else
  {
    PRCNURBSCurve *center_curve = new PRCNURBSCurve;
    center_curve->is_rational = false;
    center_curve->degree = 3;
    const uint32_t CENTER_NUMBER_OF_POINTS = n;
    center_curve->control_point.resize(CENTER_NUMBER_OF_POINTS);
    for(uint32_t i = 0; i < CENTER_NUMBER_OF_POINTS; ++i)
      center_curve->control_point[i].Set(cP[i][0],cP[i][1],cP[i][2]);
    center_curve->knot.resize(3+CENTER_NUMBER_OF_POINTS+1);
    center_curve->knot[0] = 1;
    for(uint32_t i = 1; i < 3+CENTER_NUMBER_OF_POINTS; ++i)
      center_curve->knot[i] = (i+2)/3; // integer division is intentional
    center_curve->knot[3+CENTER_NUMBER_OF_POINTS] = (3+CENTER_NUMBER_OF_POINTS+1)/3;
    surface->center_curve = center_curve;

    PRCNURBSCurve *origin_curve = new PRCNURBSCurve;
    origin_curve->is_rational = false;
    origin_curve->degree = 3;
    const uint32_t ORIGIN_NUMBER_OF_POINTS = n;
    origin_curve->control_point.resize(ORIGIN_NUMBER_OF_POINTS);
    for(uint32_t i = 0; i < ORIGIN_NUMBER_OF_POINTS; ++i)
      origin_curve->control_point[i].Set(oP[i][0],oP[i][1],oP[i][2]);
    origin_curve->knot.resize(3+ORIGIN_NUMBER_OF_POINTS+1);
    origin_curve->knot[0] = 1;
    for(size_t i = 1; i < 3+ORIGIN_NUMBER_OF_POINTS; ++i)
      origin_curve->knot[i] = (i+2)/3; // integer division is intentional
    origin_curve->knot[3+ORIGIN_NUMBER_OF_POINTS] = (3+ORIGIN_NUMBER_OF_POINTS+1)/3;
    surface->origin_curve = origin_curve;

    surface->uv_domain.min.x = 0;
    surface->uv_domain.max.x = 2*pi;
    surface->uv_domain.min.y = 1; // first knot
    surface->uv_domain.max.y = (3+CENTER_NUMBER_OF_POINTS+1)/3; // last knot
  }
}

void oPRCFile::addHemisphere(double radius, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCSphere)
  SETTRANSF
  surface->uv_domain.min.x = 0;
  surface->uv_domain.max.x = 2*pi;
  surface->uv_domain.min.y = 0;
  surface->uv_domain.max.y = 0.5*pi;
  surface->radius = radius;
}

void oPRCFile::addSphere(double radius, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCSphere)
  SETTRANSF
  surface->uv_domain.min.x = 0;
  surface->uv_domain.max.x = 2*pi;
  surface->uv_domain.min.y =-0.5*pi;
  surface->uv_domain.max.y = 0.5*pi;
  surface->radius = radius;
}

void oPRCFile::addDisk(double radius, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCRuled)
  SETTRANSF
  PRCCircle *first_curve = new PRCCircle;
  first_curve->radius = radius;
  surface->first_curve = first_curve;
  PRCCircle *second_curve = new PRCCircle;
  second_curve->radius = 0;
  surface->second_curve = second_curve;

  surface->uv_domain.min.x = 0;
  surface->uv_domain.max.x = 1;
  surface->uv_domain.min.y = 0;
  surface->uv_domain.max.y = 2*pi;
  surface->parameterization_on_v_coeff_a = -1;
  surface->parameterization_on_v_coeff_b = 2*pi;
}

void oPRCFile::addCylinder(double radius, double height, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCCylinder)
  SETTRANSF
  surface->uv_domain.min.x = 0;
  surface->uv_domain.max.x = 2*pi;
  surface->uv_domain.min.y = (height>0)?0:height;
  surface->uv_domain.max.y = (height>0)?height:0;
  surface->radius = radius;
}

void oPRCFile::addTorus(double major_radius, double minor_radius, double angle1, double angle2, const PRCmaterial &m, PRCFACETRANSFORM)
{
  ADDFACE(PRCTorus)
  SETTRANSF
  surface->uv_domain.min.x = (angle1/180)*pi;
  surface->uv_domain.max.x = (angle2/180)*pi;
  surface->uv_domain.min.y = 0;
  surface->uv_domain.max.y = 2*pi;
  surface->major_radius = major_radius;
  surface->minor_radius = minor_radius;
}

#undef PRCFACETRANSFORM
#undef ADDFACE
#undef ADDWIRE
#undef SETTRANSF

uint32_t PRCFileStructure::addMaterialGeneric(PRCMaterialGeneric*& pMaterialGeneric)
{
  globals.materials.push_back(pMaterialGeneric);
  pMaterialGeneric = NULL;
  return globals.materials.size()-1;
}

uint32_t PRCFileStructure::addTextureApplication(PRCTextureApplication*& pTextureApplication)
{
  globals.materials.push_back(pTextureApplication);
  pTextureApplication = NULL;
  return globals.materials.size()-1;
}

uint32_t PRCFileStructure::addStyle(PRCStyle*& pStyle)
{
  globals.styles.push_back(pStyle);
  pStyle = NULL;
  return globals.styles.size()-1;
}

uint32_t PRCFileStructure::addPartDefinition(PRCPartDefinition*& pPartDefinition)
{
  tree.part_definitions.push_back(pPartDefinition);
  pPartDefinition = NULL;  
  return tree.part_definitions.size()-1;
}

uint32_t PRCFileStructure::addProductOccurrence(PRCProductOccurrence*& pProductOccurrence)
{
  tree.product_occurrences.push_back(pProductOccurrence);
  pProductOccurrence = NULL;
  return tree.product_occurrences.size()-1;
}

uint32_t PRCFileStructure::addTopoContext(PRCTopoContext*& pTopoContext)
{
  contexts.push_back(pTopoContext);
  pTopoContext = NULL;
  return contexts.size()-1;
}

uint32_t PRCFileStructure::getTopoContext(PRCTopoContext*& pTopoContext)
{
  pTopoContext = new PRCTopoContext;
  contexts.push_back(pTopoContext);
  return contexts.size()-1;
}

uint32_t PRCFileStructure::add3DTess(PRC3DTess*& p3DTess)
{
  tessellations.tessellations.push_back(p3DTess);
  p3DTess = NULL;
  return tessellations.tessellations.size()-1;
}

uint32_t PRCFileStructure::add3DWireTess(PRC3DWireTess*& p3DWireTess)
{
  tessellations.tessellations.push_back(p3DWireTess);
  p3DWireTess = NULL;
  return tessellations.tessellations.size()-1;
}
/*
uint32_t PRCFileStructure::addMarkupTess(PRCMarkupTess*& pMarkupTess)
{
  tessellations.tessellations.push_back(pMarkupTess);
  pMarkupTess = NULL;
  return tessellations.tessellations.size()-1;
}

uint32_t PRCFileStructure::addMarkup(PRCMarkup*& pMarkup)
{
  tree.markups.push_back(pMarkup);
  pMarkup = NULL;
  return tree.markups.size()-1;
}

uint32_t PRCFileStructure::addAnnotationItem(PRCAnnotationItem*& pAnnotationItem)
{
  tree.annotation_entities.push_back(pAnnotationItem);
  pAnnotationItem = NULL;
  return tree.annotation_entities.size()-1;
}
*/
uint32_t PRCFileStructure::addCoordinateSystem(PRCCoordinateSystem*& pCoordinateSystem)
{
  globals.reference_coordinate_systems.push_back(pCoordinateSystem);
  pCoordinateSystem = NULL;
  return globals.reference_coordinate_systems.size()-1;
}

uint32_t PRCFileStructure::addCoordinateSystemUnique(PRCCoordinateSystem*& pCoordinateSystem)
{
  for(uint32_t i = 0; i < globals.reference_coordinate_systems.size(); ++i)
  {
    if(*(globals.reference_coordinate_systems[i])==*pCoordinateSystem) {
      pCoordinateSystem = NULL;
      return i;
    }
  }
  globals.reference_coordinate_systems.push_back(pCoordinateSystem);
  pCoordinateSystem = NULL;
  return globals.reference_coordinate_systems.size()-1;
}

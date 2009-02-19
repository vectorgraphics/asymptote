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

#ifndef __DESCRIBE_PRC_H
#define __DESCRIBE_PRC_H

#include "iPRCFile.h"
#include "bitData.h"

void describeGlobals(BitByBitData&);
void describeTree(BitByBitData&);
void describeTessellation(BitByBitData&);
void describeGeometry(BitByBitData&);
void describeExtraGeometry(BitByBitData&);
void describeModelFileData(BitByBitData&,unsigned int);

void describePicture(BitByBitData&);
void describeTextureDefinition(BitByBitData&);
void describeMaterial(BitByBitData&);
void describeLinePattern(BitByBitData&);
void describeCategory1LineStyle(BitByBitData&);
void describeFillPattern(BitByBitData&);
void describeRepresentationItem(BitByBitData&);

//void describe(BitByBitData&);
void describeLight(BitByBitData&);
void describeCamera(BitByBitData&);
bool describeContentCurve(BitByBitData&);
void describeCurvCircle(BitByBitData&);
void describeCurvLine(BitByBitData&);
void describeCurvNURBS(BitByBitData&);
void describeCurvPolyLine(BitByBitData&);
void describeContentWireEdge(BitByBitData&);
bool isCompressedSerialType(unsigned int);
void describeUVParametrization(BitByBitData&);
void describeSurfNURBS(BitByBitData&);
void describeSurfCylinder(BitByBitData&);
void describeSurfPlane(BitByBitData&);
void describeTopoFace(BitByBitData&);
void describeTopoLoop(BitByBitData&);
void describeTopoCoEdge(BitByBitData&);
void describeTopoEdge(BitByBitData&);
void describeTopoConnex(BitByBitData&);
void describeTopoShell(BitByBitData&);
void describeObject(BitByBitData&);
void describeBaseTopology(BitByBitData&);
void describeBaseGeometry(BitByBitData&);
unsigned int describeContentBody(BitByBitData&);
void describeContentSurface(BitByBitData&);
void describeBody(BitByBitData&);
void describeTopoContext(BitByBitData&);
void describeLineAttr(BitByBitData&);
void describeArrayRGBA(BitByBitData&,int,int);
void describeContentBaseTessData(BitByBitData&);
void describeTessFace(BitByBitData&);
void describe3DTess(BitByBitData&);
void describe3DWireTess(BitByBitData&);
void describe3DMarkupTess(BitByBitData&);
void describeHighlyCompressed3DTess(BitByBitData&);
void describeSceneDisplayParameters(BitByBitData&);
void describeCartesionTransformation3d(BitByBitData&);
void describeTransformation3d(BitByBitData&);
void describeTransformation2d(BitByBitData&);
void describeFileStructureInternalData(BitByBitData&);
void describeProductOccurrence(BitByBitData&);
void describeRepresentationItemContent(BitByBitData&);
void describeMarkups(BitByBitData&);
void describeAnnotationView(BitByBitData&);
void describeExtent3d(BitByBitData&);
void describeExtent2d(BitByBitData&);
void describeExtent1d(BitByBitData&);
void describeVector3d(BitByBitData&);
void describeVector2d(BitByBitData&);
void describeContentPRCBaseWithGraphics(BitByBitData&,bool);
void describeGraphics(BitByBitData&);
void describePartDefinition(BitByBitData&);
void describeRGBColour(BitByBitData&);
void describeSchema(BitByBitData&);
void describeName(BitByBitData&);
void describeAttributes(BitByBitData&);
void describeContentPRCBase(BitByBitData&,bool);
void describeUnit(BitByBitData&);
void describeCompressedUniqueID(BitByBitData&);
void describeUserData(BitByBitData&);

extern std::string currentName;
extern int layer_index;
extern int index_of_line_style;
extern unsigned short behaviour_bit_field;

void unFlushSerialization();
void resetCurrentGraphics();

bool checkSectionCode(BitByBitData&,unsigned int);

std::string getIndent();
void indent();
void dedent();

#endif // __DESCRIBE_PRC_H

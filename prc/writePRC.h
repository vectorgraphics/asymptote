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
#include "PRCbitStream.h"
#include "PRC.h"

struct Point3d
{
  double x,y,z;
  Point3d(double a, double b, double c) : x(a),y(b),z(c) {}
  void write(PRCbitStream &out)
  {
    out << x << y << z;
  }
};

struct Extent3d
{
  Point3d min,max;
  Extent3d(const Point3d &m1, const Point3d& m2) : min(m1),max(m2) {}
  void write(PRCbitStream &out)
  {
    // flip the order? Following the documentation, in a real file, min>max.
    // Considering the file to be right, min and max can be flipped in both places (here and in description)
    // resulting in no practical difference.
    // But in other places (Extent3D and Extent1D), min is really before max, so for consistency
    // this is left alone.
    min.write(out);
    max.write(out);
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
  private:
    Attributes *attributes;
    std::string name;
    bool eligibleForReference;
    uint32_t CADID, CADPersistentID, PRCID;
};

extern AttributeTitle EMPTY_ATTRIBUTE_TITLE;
extern Attribute EMPTY_ATTRIBUTE;
extern Attributes EMPTY_ATTRIBUTES;
extern ContentPRCBase EMPTY_CONTENTPRCBASE;
extern ContentPRCBase EMPTY_CONTENTPRCBASE_WITH_REFERENCE;

extern std::string currentName;
void writeName(PRCbitStream&,const std::string&);
void resetName();

extern uint32_t layer_index;
extern uint32_t index_of_line_style;
extern uint32_t behaviour_bit_field;
static const uint32_t m1=(uint32_t)-1;

void writeGraphics(PRCbitStream&,uint32_t=m1,uint32_t=m1,uint32_t=1,bool=false);
void resetGraphics();

void resetGraphicsAndName();



#endif //__WRITE_PRC_H

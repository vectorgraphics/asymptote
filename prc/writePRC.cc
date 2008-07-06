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

#include "writePRC.h"

void UserData::write(PRCbitStream &pbs)
{
  pbs << size;
  if(size > 0)
  {
    uint32_t i = 0;
    for(; i < size/8; ++i)
    {
      pbs << data[i];
    }
    if(size % 8 != 0)
    {
      for(uint32_t j = 0; j < size%8; ++j) // 0-based, big endian bit counting
      {
        pbs << (bool)(data[i] & (0x80 >> j));
      }
    }
  }
}

void SingleAttribute::write(PRCbitStream &pbs)
{
  pbs << titleIsInteger;
  if(titleIsInteger)
    pbs << title.integer;
  else
    pbs << title.text;
  pbs << type;
  switch(type)
  {
    case KEPRCModellerAttributeTypeInt:
      pbs << data.integer;
      break;
    case KEPRCModellerAttributeTypeReal:
      pbs << data.real;
      break;
    case KEPRCModellerAttributeTypeTime:
      pbs << data.time;
      break;
    case KEPRCModellerAttributeTypeString:
      pbs << data.text;
      break;
    default:
      break;
  }
}

void Attribute::write(PRCbitStream &pbs)
{
  pbs << (uint32_t)PRC_TYPE_MISC_Attribute;
  pbs << titleIsInteger;
  if(titleIsInteger)
    pbs << title.integer;
  else
    pbs << title.text;
  pbs << sizeOfAttributeKeys;
  for(uint32_t i = 0; i < sizeOfAttributeKeys; ++i)
  {
    singleAttributes[i].write(pbs);
  }
}

void Attributes::write(PRCbitStream &pbs)
{
  pbs << numberOfAttributes;
  for(uint32_t i = 0; i < numberOfAttributes; ++i)
  {
    attributes[i].write(pbs);
  }
}

void ContentPRCBase::write(PRCbitStream &pbs)
{
  attributes->write(pbs);
  writeName(pbs,name);
  if(eligibleForReference)
  {
    pbs << CADID << CADPersistentID << PRCID;
  }
}

AttributeTitle EMPTY_ATTRIBUTE_TITLE = {(char*)""};
Attribute EMPTY_ATTRIBUTE(false,EMPTY_ATTRIBUTE_TITLE,0,NULL);
Attributes EMPTY_ATTRIBUTES(0,0);
ContentPRCBase EMPTY_CONTENTPRCBASE(&EMPTY_ATTRIBUTES);

std::string currentName;
void writeName(PRCbitStream &pbs,const std::string &name)
{
  pbs << (name == currentName);
  if(name != currentName)
  {
    pbs << name;
    currentName = name;
  }
}

void resetName()
{
  currentName = "";
}

uint32_t layer_index = m1;
uint32_t index_of_line_style = m1;
uint32_t behaviour_bit_field = 1;

void writeGraphics(PRCbitStream &pbs,uint32_t l,uint32_t i,uint32_t b,bool force)
{
  if(force || layer_index != l || index_of_line_style != i || behaviour_bit_field != b)
  {
    pbs << false;
    pbs << (uint32_t)(l+1) << (uint32_t)(i+1)
         << (uint8_t)(b&0xFF) << (uint8_t)((b>>8)&0xFF);
    layer_index = l;
    index_of_line_style = i;
    behaviour_bit_field = b;
  }
  else
    pbs << true;
}

void resetGraphics()
{
  layer_index = m1;
  index_of_line_style = m1;
  behaviour_bit_field = 1;
}

void resetGraphicsAndName()
{
  resetGraphics(); resetName();
}

uint32_t makeCADID()
{
  static uint32_t ID = 1;
  return ID++;
}

uint32_t makePRCID()
{
  static uint32_t ID = 1;
  return ID++;
}

void writeUnit(PRCbitStream &out,bool fromCAD,double unit)
{
  out << fromCAD << unit;
}

void writeEmptyMarkups(PRCbitStream &out)
{
  out << (uint32_t)0 // # of linked items
      << (uint32_t)0 // # of leaders
      << (uint32_t)0 // # of markups
      << (uint32_t)0; // # of annotation entities
}

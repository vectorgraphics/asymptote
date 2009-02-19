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

#ifndef __READPRC_H
#define __READPRC_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "../PRC.h"
#include "inflation.h"

struct FileStructureInformation
{
  unsigned int UUID[4];
  unsigned int reserved;
  std::vector<unsigned int> offsets;
};


const int GLOBALS_SECTION = 0;
const int TREE_SECTION = 1;
const int TESSELLATION_SECTION = 2;
const int GEOMETRY_SECTION = 3;
const int EXTRA_GEOMETRY_SECTION = 4;

struct FileStructure
{
  unsigned int readVersion;
  unsigned int authoringVersion;
  unsigned int fileUUID[4];
  unsigned int appUUID[4];
  char* sections[5];
  unsigned int sectionLengths[5];
};

class iPRCFile
{
  public:
    iPRCFile(std::istream&);

    ~iPRCFile()
    {
      for(unsigned int i = 0; i < fileStructures.size(); ++i)
        for(unsigned int j = 0; j < 5; ++j)
          if(fileStructures[i].sections[j] != NULL)
            free(fileStructures[i].sections[j]);
      if(modelFileData!=NULL)
        free(modelFileData);
      if(buffer != 0)
        delete[] buffer;
    }

    void describe();
    void dumpSections(std::string);

  private:
    // header data
    std::vector<FileStructureInformation> fileStructureInfos;
    std::vector<FileStructure> fileStructures;
    unsigned int modelFileOffset;
    char* modelFileData;
    unsigned int modelFileLength;
    char *buffer;
    unsigned int fileSize;
    unsigned int numberOfUncompressedFiles;
};
#endif // __READPRC_H

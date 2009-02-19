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

#include "bitData.h"
#include "iPRCFile.h"
#include "describePRC.h"

using std::vector; using std::istream; using std::ios;
using std::cout; using std::endl; using std::cerr;
using std::string;

using std::ofstream;
using std::ostringstream;

void iPRCFile::dumpSections(string prefix)
{
  ofstream out;

  for(unsigned int i = 0; i < fileStructures.size(); ++i)
  {
    ostringstream name;
    name << prefix << "Structure" << i;

    out.open((name.str()+"-Globals.bin").c_str());
    out.write(fileStructures[i].sections[GLOBALS_SECTION],fileStructures[i].sectionLengths[GLOBALS_SECTION]);
    out.close();

    out.open((name.str()+"-Tree.bin").c_str());
    out.write(fileStructures[i].sections[TREE_SECTION],fileStructures[i].sectionLengths[TREE_SECTION]);
    out.close();

    out.open((name.str()+"-Tessellation.bin").c_str());
    out.write(fileStructures[i].sections[TESSELLATION_SECTION],fileStructures[i].sectionLengths[TESSELLATION_SECTION]);
    out.close();

    out.open((name.str()+"-Geometry.bin").c_str());
    out.write(fileStructures[i].sections[GEOMETRY_SECTION],fileStructures[i].sectionLengths[GEOMETRY_SECTION]);
    out.close();

    out.open((name.str()+"-ExtraGeometry.bin").c_str());
    out.write(fileStructures[i].sections[EXTRA_GEOMETRY_SECTION],fileStructures[i].sectionLengths[EXTRA_GEOMETRY_SECTION]);
    out.close();
  }
  out.open((prefix+"-ModelFile.bin").c_str());
  out.write(modelFileData,modelFileLength);
  out.close();
}

void iPRCFile::describe()
{
  /*
  for(int i = 0; i < modelFileLength; ++i)
  {
    cout << ' ' << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(static_cast<unsigned char>(mfd.readChar()));
    if(i%16 == 15)
      cout << endl;
  }
  cout << endl;
  */

  unFlushSerialization();
  for(unsigned int i = 0; i < fileStructures.size(); ++i)
  {
    cout << "File Structure " << i << ":" << endl;

    //describe header
    char *header = buffer + fileStructureInfos[i].offsets[0];
    cout << "--Header Section--" << endl;
    cout << "  Signature " << header[0] << header[1] << header[2] << endl;
    cout << "  Minimal version for read " << *(unsigned int*)(header+3) << endl;
    cout << "  Authoring version " << *(unsigned int*)(header+7) << endl;
    cout << std::hex;
    cout << "  File structure UUID " << *(unsigned int*)(header+11) << ' ' << *(unsigned int*)(header+15) << ' ' 
        << *(unsigned int*)(header+19) << ' ' << *(unsigned int*)(header+23) << endl;
    cout << "  Application UUID " << *(unsigned int*)(header+27) << ' ' << *(unsigned int*)(header+31) << ' ' 
        << *(unsigned int*)(header+35) << ' ' << *(unsigned int*)(header+39) << endl;
    cout << std::dec;
    // uncompressed files
    unsigned int numberOfUncompressedFiles = *(unsigned int*)(header+43);
    cout << "Number of uncompressed files " << numberOfUncompressedFiles << endl;
    char *position = header+47;
    for(unsigned int j = 0; j < numberOfUncompressedFiles; ++j)
    {
      cout << "Uncompressed file " << j << ":" << endl;
      unsigned int size = *(unsigned int*)position;
      cout << "  size " << size << " bytes" << endl;
      position += size+sizeof(unsigned int);
    }

    BitByBitData fileStruct(fileStructures[i].sections[GLOBALS_SECTION],fileStructures[i].sectionLengths[GLOBALS_SECTION]);
    describeSchema(fileStruct);
    describeGlobals(fileStruct);
    unFlushSerialization();

    fileStruct = BitByBitData(fileStructures[i].sections[TREE_SECTION],fileStructures[i].sectionLengths[TREE_SECTION]);
    describeTree(fileStruct);
    unFlushSerialization();

    fileStruct = BitByBitData(fileStructures[i].sections[TESSELLATION_SECTION],fileStructures[i].sectionLengths[TESSELLATION_SECTION]);
    describeTessellation(fileStruct);
    unFlushSerialization();

    fileStruct = BitByBitData(fileStructures[i].sections[GEOMETRY_SECTION],fileStructures[i].sectionLengths[GEOMETRY_SECTION]);
    describeGeometry(fileStruct);
    unFlushSerialization();

    fileStruct = BitByBitData(fileStructures[i].sections[EXTRA_GEOMETRY_SECTION],fileStructures[i].sectionLengths[EXTRA_GEOMETRY_SECTION]);
    describeExtraGeometry(fileStruct);
    unFlushSerialization();
  }

  BitByBitData mfd(modelFileData,modelFileLength);

  describeSchema(mfd);
  describeModelFileData(mfd,fileStructures.size());
  unFlushSerialization();
}

iPRCFile::iPRCFile(istream& in)
{
  char PRC[3];
  in.read(PRC,3);
  if(PRC[0] != 'P' || PRC[1] != 'R' || PRC[2] != 'C')
  {
    cerr << "Error: Invalid file format: PRC not found." << endl;
  }
  unsigned int versionForRead,authoringVersion;
  in.read((char*)&versionForRead,sizeof(versionForRead));
  in.read((char*)&authoringVersion,sizeof(authoringVersion));
  cout << "Version for reading " << versionForRead << endl;
  cout << "Authoring version " << authoringVersion << endl;
  unsigned int fileStructureUUID[4];
  in.read((char*)fileStructureUUID,sizeof(fileStructureUUID));
  //(void*) is for formatting
  cout << "File structure UUID " << (void*)(fileStructureUUID[0]) << ' ' << (void*)(fileStructureUUID[1]) << ' '
      << (void*)(fileStructureUUID[2]) << ' ' << (void*)(fileStructureUUID[3]) << endl;
  unsigned int applicationUUID[4];
  in.read((char*)applicationUUID,sizeof(applicationUUID));
  cout << "Application UUID " << (void*)(applicationUUID[0]) << ' ' << (void*)(applicationUUID[1]) << ' '
      << (void*)(applicationUUID[2]) << ' ' << (void*)(applicationUUID[3]) << endl;
  unsigned int numberOfFileStructures;
  in.read((char*)&numberOfFileStructures,sizeof(numberOfFileStructures));
  cout << "number of file structures " << numberOfFileStructures << endl;

  // load fileStructureInformation
  for(unsigned int fsi = 0; fsi < numberOfFileStructures; ++fsi)
  {
    FileStructureInformation info;
    in.read((char*)&info.UUID,sizeof(info.UUID));
    cout << "\tFile structure UUID " << (void*)(info.UUID[0]) << ' ' << (void*)(info.UUID[1]) << ' '
        << (void*)(info.UUID[2]) << ' ' << (void*)(info.UUID[3]) << endl;

    in.read((char*)&info.reserved,sizeof(info.reserved));
    cout << "\tReserved " << info.reserved << endl;

    unsigned int numberOfOffsets;
    in.read((char*)&numberOfOffsets,sizeof(numberOfOffsets));
    cout << "\tNumber of Offsets " << numberOfOffsets << endl;

    for(unsigned int oi = 0; oi < numberOfOffsets; ++oi)
    {
      unsigned int offset;
      in.read((char*)&offset,sizeof(offset));
      info.offsets.push_back(offset);
      cout << "\t\tOffset " << offset << endl;
    }
    fileStructureInfos.push_back(info);
  }
  in.read((char*)&modelFileOffset,sizeof(modelFileOffset));
  cout << "Model file offset " << modelFileOffset << endl;
  in.read((char*)&fileSize,sizeof(fileSize)); // this is not documented
  cout << "File size " << fileSize << endl;

  in.read((char*)&numberOfUncompressedFiles,sizeof(numberOfUncompressedFiles));
  cout << "Number of uncompressed files " << numberOfUncompressedFiles << endl;
  for(unsigned int ufi = 0; ufi < numberOfUncompressedFiles; ++ufi)
  {
    unsigned int size;
    in.read((char*)&size,sizeof(size));
    in.seekg(size,ios::cur);
  }

  //read the whole file into memory
  in.seekg(0,ios::beg);
  buffer = new char[fileSize];
  if(!buffer) cerr << "Couldn't get memory." << endl;
  in.read(buffer,fileSize);
  //decompress fileStructures
  for(unsigned int fs = 0; fs < fileStructureInfos.size(); ++fs)
  {
    fileStructures.push_back(FileStructure());
    for(unsigned int i = 1; i < fileStructureInfos[fs].offsets.size(); ++i) // start at 1 since header is decompressed
    {
      fileStructures[fs].sections[i-1] = NULL;
      unsigned int offset = fileStructureInfos[fs].offsets[i];
      fileStructures[fs].sectionLengths[i-1] = decompress(buffer+offset,fileSize-offset,fileStructures[fs].sections[i-1]);
    }
  }

  //decompress modelFileData
  modelFileData = NULL;
  modelFileLength = decompress(buffer+modelFileOffset,fileSize-modelFileOffset,modelFileData);
}

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
#include "bitData.h"

using namespace std;

int main(int argc, char *argv[])
{
  if(argc < 2)
  {
    cerr << "Error: Input file not specified." << endl;
    return 1;
  }
  ifstream inFile(argv[1]);
  if(!inFile)
  {
    cerr << "Error: Cannot open input file." << endl;
    return 1;
  }
  inFile.seekg(0,ios::end);
  unsigned int length = inFile.tellg();
  inFile.seekg(0,ios::beg);

  char *buf = new char[length];

  inFile.read(buf,length);
  BitByBitData bbbd(buf,length);

  double dsf;
  cout << "double to search for: "; cin >> dsf;
  BitPosition currP;
  for(currP.byteIndex = 0; currP.byteIndex < length; ++currP.byteIndex)
    for(currP.bitIndex = 0; currP.bitIndex < 8; ++currP.bitIndex)
  {
    bbbd.setPosition(currP);
    if(bbbd.readDouble() == dsf)
    {
      BitPosition bp = bbbd.getPosition();
      cout << "Found " << dsf << " at " << currP.byteIndex << ':'
          << currP.bitIndex << " to " << bp.byteIndex << ':' << bp.bitIndex
          << endl;
    }
  }
  delete[] buf;
  return 0;
}

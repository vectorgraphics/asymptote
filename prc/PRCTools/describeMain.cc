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
#include "iPRCFile.h"

using namespace std;

int main(int argc, char* argv[])
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

  iPRCFile myFile(inFile);
  myFile.describe();

  return 0;
}

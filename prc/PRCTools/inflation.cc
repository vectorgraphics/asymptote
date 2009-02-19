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

#include "inflation.h"

using std::istream;
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::exit;

int decompress(char* inb, int fileLength, char* &outb)
{
  const int CHUNK = 16384;
  unsigned int resultSize = 0;

  outb = (char*) realloc(outb,CHUNK);
  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.avail_in = fileLength;
  strm.next_in = (unsigned char*)inb;
  strm.opaque = Z_NULL;
  int code = inflateInit(&strm);

  if(code != Z_OK)
    return -1;

  strm.next_out = (unsigned char*)outb;
  strm.avail_out = CHUNK;
  code = inflate(&strm,Z_NO_FLUSH);
  resultSize = CHUNK-strm.avail_out;

  unsigned int size = CHUNK;
  while(code == Z_OK)
  {
    outb = (char*) realloc(outb,2*size);
    if(outb == NULL)
    {
      cerr << "Ran out of memory while decompressing." << endl;
      exit(1);
    }
    strm.next_out = (Bytef*)(outb + resultSize);
    strm.avail_out += size;
    size *= 2;
    code = inflate(&strm,Z_NO_FLUSH);
    resultSize = size - strm.avail_out;
  }

  code = inflateEnd(&strm);
  if(code != Z_OK)
  {
    free(outb);
    return 0;
  }

  return resultSize;
}

int decompress(istream &input,char* &result)
{
  input.seekg(0,ios::end);
  int fileLength = input.tellg();
  input.seekg(0,ios::beg);

  char *inb = new char[fileLength];
  input.read(inb,fileLength);

  int code = decompress(inb,fileLength,result);
  delete[] inb;
  return code;
}

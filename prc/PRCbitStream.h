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

#ifndef __PRC_BIT_STREAM_H
#define __PRC_BIT_STREAM_H

#include <string>
#include <stdlib.h>
#include <stdint.h>

#define CHUNK_SIZE (1024)
// Is this a reasonable initial size?

class PRCbitStream
{
  public:
    PRCbitStream(uint8_t*& buff, unsigned int l) : byteIndex(0), bitIndex(0),
                 allocatedLength(l), data(buff), compressed(false)
    {
      if(data == 0)
      {
        getAChunk();
      }
    }

    unsigned int getSize() const;
    uint8_t* getData();

    PRCbitStream& operator <<(const std::string&);
    PRCbitStream& operator <<(bool);
    PRCbitStream& operator <<(uint32_t);
    PRCbitStream& operator <<(uint8_t);
    PRCbitStream& operator <<(int32_t);
    PRCbitStream& operator <<(double);
    PRCbitStream& operator <<(const char*);

    void compress();
  private:
    void writeBit(bool);
    void writeBits(uint32_t,uint8_t);
    void writeByte(uint8_t);
    void nextByte();
    void nextBit();
    void getAChunk();
    // bitIndex is "big endian", zero based, location of next bit to write
    unsigned int byteIndex,bitIndex;
    unsigned int allocatedLength;
    uint8_t*& data;
    bool compressed;
    uint32_t compressedDataSize;
};

#endif // __PRC_BIT_STREAM_H

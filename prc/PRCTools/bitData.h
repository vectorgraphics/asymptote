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

#ifndef __BITDATA_H
#define __BITDATA_H

#include <iostream>
#include <string>

struct BitPosition
{
  unsigned int byteIndex;
  unsigned int bitIndex;
};

class BitByBitData
{
  public:
    BitByBitData(char* s,unsigned int l) : start(s),data(s),length(l),
                 bitMask(0x80),showBits(false),failed(false) {}

    void tellPosition();
    BitPosition getPosition();
    void setPosition(const BitPosition&);
    void setPosition(unsigned int,unsigned int);
    void setShowBits(bool);
    bool readBit();
    unsigned char readChar();

    unsigned int readUnsignedInt();
    std::string readString();
    int readInt();
    double readDouble();

  private:
    char *start;  // first byte so we know where we are
    char *data;    // last byte read
    unsigned int length;
    unsigned char bitMask;  // mask to read next bit of current byte
    bool showBits; // show each bit read?
    bool failed;
    void nextBit(); // shift bit mask and get next byte if needed
};

#endif // __BITDATA_H

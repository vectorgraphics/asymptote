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

#include "../PRC.h"
#include "../PRCdouble.h"
#include "bitData.h"

using std::cout; using std::endl; using std::hex; using std::cerr;

BitPosition BitByBitData::getPosition()
{
  BitPosition bp;
  bp.byteIndex = data - start;
  bp.bitIndex = 0;
  for(unsigned char temp = bitMask<<1; temp != 0; temp <<= 1)
    bp.bitIndex++;
  return bp;
}

void BitByBitData::setPosition(const BitPosition& bp)
{
  if(bp.byteIndex < length)
  {
    data = start + bp.byteIndex;
    bitMask = 0x80 >> bp.bitIndex;
    // big-endian, zero based bit index (from 0 to 7)
    // so 0x80 => bit 0, 0x01 => bit 7
    // Why? It is easy to see in a hex editor.
    failed = false;
  }
  else
  {
    failed = true;
  }
}

void BitByBitData::setPosition(unsigned int byte, unsigned int bit)
{
  if(byte <= length)
  {
    data = start + byte;
    bitMask = 0x80 >> bit;
    // big-endian, zero based bit index (from 0 to 7)
    // so 0x80 => bit 0, 0x01 => bit 7
    // Why? It is easy to see in a hex editor.
    failed = false;
  }
  else
  {
    failed = true;
  }
}

void BitByBitData::setShowBits(bool val)
{
  showBits = val;
}

void BitByBitData::tellPosition()
{
  BitPosition bp = getPosition();
  cout << bp.byteIndex << ':' << bp.bitIndex << endl;
}

bool BitByBitData::readBit()
{
  if(!failed)
  {
    bool val = *data & bitMask;
    if(showBits) cout << (val?'1':'0');
    nextBit();
    return val;
  }
  else
    return false;
}

unsigned char BitByBitData::readChar()
{
  unsigned char dat = 0;
  dat |= readBit();
  for(int a = 0; a < 7; ++a)
  {
    dat <<= 1;
    dat |= readBit();
  }
  return dat;
}

unsigned int BitByBitData::readUnsignedInt()
{
  unsigned int result = 0;
  unsigned int count = 0;
  while(readBit())
  {
    result |= (static_cast<unsigned int>(readChar()) << 8*count++);
  }
  if(showBits) cout << " " << result << endl;
  return result;
}

std::string BitByBitData::readString()
{
  bool isNotNull = readBit();
  std::string result;
  if(isNotNull)
  {
    unsigned int stringLength = readUnsignedInt();
    char *buf = new char[stringLength+1];
    buf[stringLength] = '\0';
    for(unsigned int a = 0; a < stringLength; ++a)
    {
      buf[a] = readChar();
    }
    result = buf;
    delete[] buf;
  }
  if(showBits) cout << " " << result << endl;
  return result;
}

int BitByBitData::readInt()
{
  int result = 0;
  unsigned int count = 0;
  while(readBit())
  {
    result |= (static_cast<unsigned int>(readChar()) << 8*count++);
  }
  result <<= (4-count)*8;
  result >>= (4-count)*8;
  if(showBits) cout << " " << result << endl;
  return result;
}


// Thanks to Michail Vidiassov
double BitByBitData::readDouble()
{
  ieee754_double value;
  value.d = 0;
  sCodageOfFrequentDoubleOrExponent *pcofdoe;
  unsigned int ucofdoe = 0;
  for(int i = 1; i <= 22; ++i)
  {
    ucofdoe <<= 1;
    ucofdoe |= readBit();
    if((pcofdoe = getcofdoe(ucofdoe,i)) != NULL)
      break;
  }
  value.d = pcofdoe->u2uod.Value;

  // check if zero
  if(pcofdoe->NumberOfBits==2 && pcofdoe->Bits==1 && pcofdoe->Type==VT_double)
    return value.d;

  value.ieee.negative = readBit(); // get sign

  if(pcofdoe->Type == VT_double) // double from list
    return value.d;

  if(readBit()==0) // no mantissa
    return value.d;

  // read the mantissa
  // read uppermost 4 bits of mantissa
  unsigned char b4 = 0;
  for(int i = 0; i < 4; ++i)
  {
    b4 <<= 1;
    b4 |= readBit();
  }

#ifdef WORDS_BIGENDIAN
  *(reinterpret_cast<unsigned char*>(&value)+1) |= b4;
  unsigned char *lastByte = reinterpret_cast<unsigned char*>(&value)+7;
  unsigned char *currentByte = reinterpret_cast<unsigned char*>(&value)+2;
#else
  *(reinterpret_cast<unsigned char*>(&value)+6) |= b4;
  unsigned char *lastByte = reinterpret_cast<unsigned char*>(&value)+0;
  unsigned char *currentByte = reinterpret_cast<unsigned char*>(&value)+5;
#endif

  for(;MOREBYTE(currentByte,lastByte); NEXTBYTE(currentByte))
  {
    if(readBit())
    {
      // new byte
      *currentByte = readChar();
    }
    else
    {
      // get 3 bit offset
      unsigned int offset = 0;
      offset |= (readBit() << 2);
      offset |= (readBit() << 1);
      offset |= readBit();
      if(offset == 0)
      {
        // fill remaining bytes in mantissa with previous byte
        unsigned char pByte = BYTEAT(currentByte,1);
        for(;MOREBYTE(currentByte,lastByte); NEXTBYTE(currentByte))
          *currentByte = pByte;
        break;
      }
      else if(offset == 6)
      {
        // fill remaining bytes except last byte with previous byte
        unsigned char pByte = BYTEAT(currentByte,1);
        PREVIOUSBYTE(lastByte);
        for(;MOREBYTE(currentByte,lastByte); NEXTBYTE(currentByte))
          *currentByte = pByte;
        *currentByte = readChar();
        break;
      }
      else
      {
        // one repeated byte
        *currentByte = BYTEAT(currentByte,offset);
      }
    }
  }
  if(showBits) cout << " " << value.d << endl;
  return value.d;
}

void BitByBitData::nextBit()
{
  bitMask >>= 1;
  if(bitMask == 0)
  {
    if(data < start+length)
      data++;
    else
    {
      failed = true;
      cout << "End of data."<< endl;
    }
    bitMask = 0x80;
  }
}

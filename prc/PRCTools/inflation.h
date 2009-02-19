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

#ifndef __INFLATION_H
#define __INFLATION_H

#include <iostream>
#include <cstdlib>
#include <zlib.h>

int decompress(std::istream&,char*&);
int decompress(char*,int,char*&);

#endif // __INFLATION_H

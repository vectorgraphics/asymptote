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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include "oPRCFile.h"

using namespace std;

int main()
{
  oPRCFile file("test.prc");

  const size_t N_COLOURS = 32;
  RGBAColour colours[N_COLOURS];
  for(size_t i = 0; i < N_COLOURS; ++i)
  {
    colours[i%N_COLOURS].R = 0.0;
    colours[i%N_COLOURS].G = (i%N_COLOURS)/static_cast<double>(N_COLOURS);
    colours[i%N_COLOURS].B = 0.95;
    colours[i%N_COLOURS].A = 0.75;
  }
  
  PRCmaterial materials[N_COLOURS];
  for(size_t i = 0; i < N_COLOURS; ++i)
  {
    materials[i%N_COLOURS].diffuse.R = 0.0;
    materials[i%N_COLOURS].diffuse.G = (i%N_COLOURS)/static_cast<double>(N_COLOURS);
    materials[i%N_COLOURS].diffuse.B = 0.95;
    materials[i%N_COLOURS].diffuse.A = 0.75;
    materials[i%N_COLOURS].specular.R = 0.01*0.0;
    materials[i%N_COLOURS].specular.G = 0.01*(i%N_COLOURS)/static_cast<double>(N_COLOURS);
    materials[i%N_COLOURS].specular.B = 0.01*0.95;
    materials[i%N_COLOURS].specular.A = 0.01*0.75;
    materials[i%N_COLOURS].emissive.R = 0.20*0.0;
    materials[i%N_COLOURS].emissive.G = 0.20*(i%N_COLOURS)/static_cast<double>(N_COLOURS);
    materials[i%N_COLOURS].emissive.B = 0.20*0.95;
    materials[i%N_COLOURS].emissive.A = 0.20*0.75;
    materials[i%N_COLOURS].ambient.R  = 0.05*0.0;
    materials[i%N_COLOURS].ambient.G  = 0.05*(i%N_COLOURS)/static_cast<double>(N_COLOURS);
    materials[i%N_COLOURS].ambient.B  = 0.05*0.95;
    materials[i%N_COLOURS].ambient.A  = 0.05*0.75;
    materials[i%N_COLOURS].alpha      = 0.75;
    materials[i%N_COLOURS].shininess  = 0.1;
  }
  
  if(1) {
  double knotsU[] = {1,1,1,1,2,2,2,2};
  double knotsV[] = {1,1,1,1,2,2,2,2};
  const size_t NUMBER_OF_PATCHES = 32;
  double controlPoints[NUMBER_OF_PATCHES][16][3] = 
  {
    { // Patch 0
      {1.4,0,2.4},{1.4,-0.784,2.4},{0.784,-1.4,2.4},{0,-1.4,2.4},
      {1.3375,0,2.53125},{1.3375,-0.749,2.53125},{0.749,-1.3375,2.53125},{0,-1.3375,2.53125},
      {1.4375,0,2.53125},{1.4375,-0.805,2.53125},{0.805,-1.4375,2.53125},{0,-1.4375,2.53125},
      {1.5,0,2.4},{1.5,-0.84,2.4},{0.84,-1.5,2.4},{0,-1.5,2.4},
    },
    { // Patch 1
      {0,-1.4,2.4},{-0.784,-1.4,2.4},{-1.4,-0.784,2.4},{-1.4,0,2.4},
      {0,-1.3375,2.53125},{-0.749,-1.3375,2.53125},{-1.3375,-0.749,2.53125},{-1.3375,0,2.53125},
      {0,-1.4375,2.53125},{-0.805,-1.4375,2.53125},{-1.4375,-0.805,2.53125},{-1.4375,0,2.53125},
      {0,-1.5,2.4},{-0.84,-1.5,2.4},{-1.5,-0.84,2.4},{-1.5,0,2.4},
    },
    { // Patch 2
      {-1.4,0,2.4},{-1.4,0.784,2.4},{-0.784,1.4,2.4},{0,1.4,2.4},
      {-1.3375,0,2.53125},{-1.3375,0.749,2.53125},{-0.749,1.3375,2.53125},{0,1.3375,2.53125},
      {-1.4375,0,2.53125},{-1.4375,0.805,2.53125},{-0.805,1.4375,2.53125},{0,1.4375,2.53125},
      {-1.5,0,2.4},{-1.5,0.84,2.4},{-0.84,1.5,2.4},{0,1.5,2.4},
    },
    { // Patch 3
      {0,1.4,2.4},{0.784,1.4,2.4},{1.4,0.784,2.4},{1.4,0,2.4},
      {0,1.3375,2.53125},{0.749,1.3375,2.53125},{1.3375,0.749,2.53125},{1.3375,0,2.53125},
      {0,1.4375,2.53125},{0.805,1.4375,2.53125},{1.4375,0.805,2.53125},{1.4375,0,2.53125},
      {0,1.5,2.4},{0.84,1.5,2.4},{1.5,0.84,2.4},{1.5,0,2.4},
    },
    { // Patch 4
      {1.5,0,2.4},{1.5,-0.84,2.4},{0.84,-1.5,2.4},{0,-1.5,2.4},
      {1.75,0,1.875},{1.75,-0.98,1.875},{0.98,-1.75,1.875},{0,-1.75,1.875},
      {2,0,1.35},{2,-1.12,1.35},{1.12,-2,1.35},{0,-2,1.35},
      {2,0,0.9},{2,-1.12,0.9},{1.12,-2,0.9},{0,-2,0.9},
    },
    { // Patch 5
      {0,-1.5,2.4},{-0.84,-1.5,2.4},{-1.5,-0.84,2.4},{-1.5,0,2.4},
      {0,-1.75,1.875},{-0.98,-1.75,1.875},{-1.75,-0.98,1.875},{-1.75,0,1.875},
      {0,-2,1.35},{-1.12,-2,1.35},{-2,-1.12,1.35},{-2,0,1.35},
      {0,-2,0.9},{-1.12,-2,0.9},{-2,-1.12,0.9},{-2,0,0.9},
    },
    { // Patch 6
      {-1.5,0,2.4},{-1.5,0.84,2.4},{-0.84,1.5,2.4},{0,1.5,2.4},
      {-1.75,0,1.875},{-1.75,0.98,1.875},{-0.98,1.75,1.875},{0,1.75,1.875},
      {-2,0,1.35},{-2,1.12,1.35},{-1.12,2,1.35},{0,2,1.35},
      {-2,0,0.9},{-2,1.12,0.9},{-1.12,2,0.9},{0,2,0.9},
    },
    { // Patch 7
      {0,1.5,2.4},{0.84,1.5,2.4},{1.5,0.84,2.4},{1.5,0,2.4},
      {0,1.75,1.875},{0.98,1.75,1.875},{1.75,0.98,1.875},{1.75,0,1.875},
      {0,2,1.35},{1.12,2,1.35},{2,1.12,1.35},{2,0,1.35},
      {0,2,0.9},{1.12,2,0.9},{2,1.12,0.9},{2,0,0.9},
    },
    { // Patch 8
      {2,0,0.9},{2,-1.12,0.9},{1.12,-2,0.9},{0,-2,0.9},
      {2,0,0.45},{2,-1.12,0.45},{1.12,-2,0.45},{0,-2,0.45},
      {1.5,0,0.225},{1.5,-0.84,0.225},{0.84,-1.5,0.225},{0,-1.5,0.225},
      {1.5,0,0.15},{1.5,-0.84,0.15},{0.84,-1.5,0.15},{0,-1.5,0.15},
    },
    { // Patch 9
      {0,-2,0.9},{-1.12,-2,0.9},{-2,-1.12,0.9},{-2,0,0.9},
      {0,-2,0.45},{-1.12,-2,0.45},{-2,-1.12,0.45},{-2,0,0.45},
      {0,-1.5,0.225},{-0.84,-1.5,0.225},{-1.5,-0.84,0.225},{-1.5,0,0.225},
      {0,-1.5,0.15},{-0.84,-1.5,0.15},{-1.5,-0.84,0.15},{-1.5,0,0.15},
    },
    { // Patch 10
      {-2,0,0.9},{-2,1.12,0.9},{-1.12,2,0.9},{0,2,0.9},
      {-2,0,0.45},{-2,1.12,0.45},{-1.12,2,0.45},{0,2,0.45},
      {-1.5,0,0.225},{-1.5,0.84,0.225},{-0.84,1.5,0.225},{0,1.5,0.225},
      {-1.5,0,0.15},{-1.5,0.84,0.15},{-0.84,1.5,0.15},{0,1.5,0.15},
    },
    { // Patch 11
      {0,2,0.9},{1.12,2,0.9},{2,1.12,0.9},{2,0,0.9},
      {0,2,0.45},{1.12,2,0.45},{2,1.12,0.45},{2,0,0.45},
      {0,1.5,0.225},{0.84,1.5,0.225},{1.5,0.84,0.225},{1.5,0,0.225},
      {0,1.5,0.15},{0.84,1.5,0.15},{1.5,0.84,0.15},{1.5,0,0.15},
    },
    { // Patch 12
      {-1.6,0,2.025},{-1.6,-0.3,2.025},{-1.5,-0.3,2.25},{-1.5,0,2.25},
      {-2.3,0,2.025},{-2.3,-0.3,2.025},{-2.5,-0.3,2.25},{-2.5,0,2.25},
      {-2.7,0,2.025},{-2.7,-0.3,2.025},{-3,-0.3,2.25},{-3,0,2.25},
      {-2.7,0,1.8},{-2.7,-0.3,1.8},{-3,-0.3,1.8},{-3,0,1.8},
    },
    { // Patch 13
      {-1.5,0,2.25},{-1.5,0.3,2.25},{-1.6,0.3,2.025},{-1.6,0,2.025},
      {-2.5,0,2.25},{-2.5,0.3,2.25},{-2.3,0.3,2.025},{-2.3,0,2.025},
      {-3,0,2.25},{-3,0.3,2.25},{-2.7,0.3,2.025},{-2.7,0,2.025},
      {-3,0,1.8},{-3,0.3,1.8},{-2.7,0.3,1.8},{-2.7,0,1.8},
    },
    { // Patch 14
      {-2.7,0,1.8},{-2.7,-0.3,1.8},{-3,-0.3,1.8},{-3,0,1.8},
      {-2.7,0,1.575},{-2.7,-0.3,1.575},{-3,-0.3,1.35},{-3,0,1.35},
      {-2.5,0,1.125},{-2.5,-0.3,1.125},{-2.65,-0.3,0.9375},{-2.65,0,0.9375},
      {-2,0,0.9},{-2,-0.3,0.9},{-1.9,-0.3,0.6},{-1.9,0,0.6},
    },
    { // Patch 15
      {-3,0,1.8},{-3,0.3,1.8},{-2.7,0.3,1.8},{-2.7,0,1.8},
      {-3,0,1.35},{-3,0.3,1.35},{-2.7,0.3,1.575},{-2.7,0,1.575},
      {-2.65,0,0.9375},{-2.65,0.3,0.9375},{-2.5,0.3,1.125},{-2.5,0,1.125},
      {-1.9,0,0.6},{-1.9,0.3,0.6},{-2,0.3,0.9},{-2,0,0.9},
    },
    { // Patch 16
      {1.7,0,1.425},{1.7,-0.66,1.425},{1.7,-0.66,0.6},{1.7,0,0.6},
      {2.6,0,1.425},{2.6,-0.66,1.425},{3.1,-0.66,0.825},{3.1,0,0.825},
      {2.3,0,2.1},{2.3,-0.25,2.1},{2.4,-0.25,2.025},{2.4,0,2.025},
      {2.7,0,2.4},{2.7,-0.25,2.4},{3.3,-0.25,2.4},{3.3,0,2.4},
    },
    { // Patch 17
      {1.7,0,0.6},{1.7,0.66,0.6},{1.7,0.66,1.425},{1.7,0,1.425},
      {3.1,0,0.825},{3.1,0.66,0.825},{2.6,0.66,1.425},{2.6,0,1.425},
      {2.4,0,2.025},{2.4,0.25,2.025},{2.3,0.25,2.1},{2.3,0,2.1},
      {3.3,0,2.4},{3.3,0.25,2.4},{2.7,0.25,2.4},{2.7,0,2.4},
    },
    { // Patch 18
      {2.7,0,2.4},{2.7,-0.25,2.4},{3.3,-0.25,2.4},{3.3,0,2.4},
      {2.8,0,2.475},{2.8,-0.25,2.475},{3.525,-0.25,2.49375},{3.525,0,2.49375},
      {2.9,0,2.475},{2.9,-0.15,2.475},{3.45,-0.15,2.5125},{3.45,0,2.5125},
      {2.8,0,2.4},{2.8,-0.15,2.4},{3.2,-0.15,2.4},{3.2,0,2.4},
    },
    { // Patch 19
      {3.3,0,2.4},{3.3,0.25,2.4},{2.7,0.25,2.4},{2.7,0,2.4},
      {3.525,0,2.49375},{3.525,0.25,2.49375},{2.8,0.25,2.475},{2.8,0,2.475},
      {3.45,0,2.5125},{3.45,0.15,2.5125},{2.9,0.15,2.475},{2.9,0,2.475},
      {3.2,0,2.4},{3.2,0.15,2.4},{2.8,0.15,2.4},{2.8,0,2.4},
    },
    { // Patch 20
      {0,0,3.15},{0,0,3.15},{0,0,3.15},{0,0,3.15},
      {0.8,0,3.15},{0.8,-0.45,3.15},{0.45,-0.8,3.15},{0,-0.8,3.15},
      {0,0,2.85},{0,0,2.85},{0,0,2.85},{0,0,2.85},
      {0.2,0,2.7},{0.2,-0.112,2.7},{0.112,-0.2,2.7},{0,-0.2,2.7},
    },
    { // Patch 21
      {0,0,3.15},{0,0,3.15},{0,0,3.15},{0,0,3.15},
      {0,-0.8,3.15},{-0.45,-0.8,3.15},{-0.8,-0.45,3.15},{-0.8,0,3.15},
      {0,0,2.85},{0,0,2.85},{0,0,2.85},{0,0,2.85},
      {0,-0.2,2.7},{-0.112,-0.2,2.7},{-0.2,-0.112,2.7},{-0.2,0,2.7},
    },
    { // Patch 22
      {0,0,3.15},{0,0,3.15},{0,0,3.15},{0,0,3.15},
      {-0.8,0,3.15},{-0.8,0.45,3.15},{-0.45,0.8,3.15},{0,0.8,3.15},
      {0,0,2.85},{0,0,2.85},{0,0,2.85},{0,0,2.85},
      {-0.2,0,2.7},{-0.2,0.112,2.7},{-0.112,0.2,2.7},{0,0.2,2.7},
    },
    { // Patch 23
      {0,0,3.15},{0,0,3.15},{0,0,3.15},{0,0,3.15},
      {0,0.8,3.15},{0.45,0.8,3.15},{0.8,0.45,3.15},{0.8,0,3.15},
      {0,0,2.85},{0,0,2.85},{0,0,2.85},{0,0,2.85},
      {0,0.2,2.7},{0.112,0.2,2.7},{0.2,0.112,2.7},{0.2,0,2.7},
    },
    { // Patch 24
      {0.2,0,2.7},{0.2,-0.112,2.7},{0.112,-0.2,2.7},{0,-0.2,2.7},
      {0.4,0,2.55},{0.4,-0.224,2.55},{0.224,-0.4,2.55},{0,-0.4,2.55},
      {1.3,0,2.55},{1.3,-0.728,2.55},{0.728,-1.3,2.55},{0,-1.3,2.55},
      {1.3,0,2.4},{1.3,-0.728,2.4},{0.728,-1.3,2.4},{0,-1.3,2.4},
    },
    { // Patch 25
      {0,-0.2,2.7},{-0.112,-0.2,2.7},{-0.2,-0.112,2.7},{-0.2,0,2.7},
      {0,-0.4,2.55},{-0.224,-0.4,2.55},{-0.4,-0.224,2.55},{-0.4,0,2.55},
      {0,-1.3,2.55},{-0.728,-1.3,2.55},{-1.3,-0.728,2.55},{-1.3,0,2.55},
      {0,-1.3,2.4},{-0.728,-1.3,2.4},{-1.3,-0.728,2.4},{-1.3,0,2.4},
    },
    { // Patch 26
      {-0.2,0,2.7},{-0.2,0.112,2.7},{-0.112,0.2,2.7},{0,0.2,2.7},
      {-0.4,0,2.55},{-0.4,0.224,2.55},{-0.224,0.4,2.55},{0,0.4,2.55},
      {-1.3,0,2.55},{-1.3,0.728,2.55},{-0.728,1.3,2.55},{0,1.3,2.55},
      {-1.3,0,2.4},{-1.3,0.728,2.4},{-0.728,1.3,2.4},{0,1.3,2.4},
    },
    { // Patch 27
      {0,0.2,2.7},{0.112,0.2,2.7},{0.2,0.112,2.7},{0.2,0,2.7},
      {0,0.4,2.55},{0.224,0.4,2.55},{0.4,0.224,2.55},{0.4,0,2.55},
      {0,1.3,2.55},{0.728,1.3,2.55},{1.3,0.728,2.55},{1.3,0,2.55},
      {0,1.3,2.4},{0.728,1.3,2.4},{1.3,0.728,2.4},{1.3,0,2.4},
    },
    { // Patch 28
      {0,0,0},{0,0,0},{0,0,0},{0,0,0},
      {1.425,0,0},{1.425,0.798,0},{0.798,1.425,0},{0,1.425,0},
      {1.5,0,0.075},{1.5,0.84,0.075},{0.84,1.5,0.075},{0,1.5,0.075},
      {1.5,0,0.15},{1.5,0.84,0.15},{0.84,1.5,0.15},{0,1.5,0.15},
    },
    { // Patch 29
      {0,0,0},{0,0,0},{0,0,0},{0,0,0},
      {0,1.425,0},{-0.798,1.425,0},{-1.425,0.798,0},{-1.425,0,0},
      {0,1.5,0.075},{-0.84,1.5,0.075},{-1.5,0.84,0.075},{-1.5,0,0.075},
      {0,1.5,0.15},{-0.84,1.5,0.15},{-1.5,0.84,0.15},{-1.5,0,0.15},
    },
    { // Patch 30
      {0,0,0},{0,0,0},{0,0,0},{0,0,0},
      {-1.425,0,0},{-1.425,-0.798,0},{-0.798,-1.425,0},{0,-1.425,0},
      {-1.5,0,0.075},{-1.5,-0.84,0.075},{-0.84,-1.5,0.075},{0,-1.5,0.075},
      {-1.5,0,0.15},{-1.5,-0.84,0.15},{-0.84,-1.5,0.15},{0,-1.5,0.15},
    },
    { // Patch 31
      {0,0,0},{0,0,0},{0,0,0},{0,0,0},
      {0,-1.425,0},{0.798,-1.425,0},{1.425,-0.798,0},{1.425,0,0},
      {0,-1.5,0.075},{0.84,-1.5,0.075},{1.5,-0.84,0.075},{1.5,0,0.075},
      {0,-1.5,0.15},{0.84,-1.5,0.15},{1.5,-0.84,0.15},{1.5,0,0.15},
    },
  };
    const size_t NUMBER_OF_TEAPOTS = 2;
    double shifted_controlPoints[NUMBER_OF_TEAPOTS][NUMBER_OF_PATCHES][16][3];
    for(size_t teapot = 0; teapot < NUMBER_OF_TEAPOTS; ++teapot)
      for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
        for(size_t i = 0; i < 16; ++i)
        {
          shifted_controlPoints[teapot][patch][i][0] = controlPoints[patch][i][0]+6*teapot;
          shifted_controlPoints[teapot][patch][i][1] = controlPoints[patch][i][1];
          shifted_controlPoints[teapot][patch][i][2] = controlPoints[patch][i][2];
        }
    file.begingroup("Teapot");
    for(size_t i = 0; i < NUMBER_OF_PATCHES; ++i)
    {
  //   was so in old API
  //   psn[i] = new PRCsurface(&file,3,3,4,4,controlPoints[i],knotsU,knotsV,colours[i%N_COLOURS]);
  //   file.add(psn[i]);
       if (1) file.addPatch(controlPoints[i],materials[i%N_COLOURS],0);
       if (0) file.addSurface(3,3,4,4,controlPoints[i],knotsU,knotsV,materials[i%N_COLOURS],NULL,0); // use (too) general API for the same result as above
    }
    file.endgroup();
    file.begingroup("Teapot rendered in the way of opaque surfacesPRCNOBREAKPRCCOMPRESSLOW");
    for(size_t i = 0; i < NUMBER_OF_PATCHES; ++i)
    {
       file.addPatch(shifted_controlPoints[1][i],materials[i%N_COLOURS],0); // force joining together of patches, damaging transparency
    }
    file.endgroup();
  }

  if(1) {
  const size_t NUMBER_OF_POINTS = 31;
  double points[NUMBER_OF_POINTS][3];
  for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
  {
    points[i][0] = 3.5*cos(3.0*i/NUMBER_OF_POINTS*2.0*M_PI);
    points[i][1] = 3.5*sin(3.0*i/NUMBER_OF_POINTS*2.0*M_PI);
    points[i][2] = 5.0*i/NUMBER_OF_POINTS-1.0;
  }
  const size_t NUMBER_OF_WIRES = 2;
  double shifted_points[NUMBER_OF_WIRES][NUMBER_OF_POINTS][3];
  for(size_t wire = 0; wire < NUMBER_OF_WIRES; ++wire)
    for(size_t point = 0; point < NUMBER_OF_POINTS; ++point)
      {
        shifted_points[wire][point][0] = points[point][0];
        shifted_points[wire][point][1] = points[point][1];
        shifted_points[wire][point][2] = points[point][2]+0.1*wire+0.1;
      }
  double knots[3+NUMBER_OF_POINTS+1];
  knots[0] = 1;
  for(size_t i = 1; i < 3+NUMBER_OF_POINTS; ++i)
  {
    knots[i] = (i+2)/3; // integer division is intentional
  }
  knots[3+NUMBER_OF_POINTS] = (3+NUMBER_OF_POINTS+1)/3;

  double point[3] = {11,0,0};
  file.begingroup("point");
  file.addPoint(point, RGBAColour(1.0,0.0,0.0));
  file.endgroup();

// RGBAColour red(1.0,0.0,0.0);
// PRCline pl(&file,NUMBER_OF_POINTS,points,red);
// file.add(&pl);
  file.begingroup("polyline");
  file.addLine(NUMBER_OF_POINTS, points, RGBAColour(1.0,0.0,0.0));
  file.endgroup();

  file.begingroup("polylines");
  file.addLine(NUMBER_OF_POINTS, shifted_points[0], RGBAColour(0.0,1.0,0.0));
  file.addLine(NUMBER_OF_POINTS, shifted_points[1], RGBAColour(1.0,1.0,0.0));
  file.endgroup();

// RGBAColour white(1.0,1.0,1.0);
// PRCcurve pc(&file,3,NUMBER_OF_POINTS,points,knots,white);
// file.add(&pc);
  if(1)
  {
    file.begingroup("bezier_wire");
    file.addBezierCurve(NUMBER_OF_POINTS,points,RGBAColour(1.0,1.0,1.0));
    file.endgroup();
  }
  if(0)
  {
    file.begingroup("NURBS_wire");
    file.addCurve(3, NUMBER_OF_POINTS, points, knots, RGBAColour(1.0,1.0,1.0), NULL); // genarl API for the above
    file.endgroup();
  }

  } 

// following box examples show a) different ways to represent a surface consisting of flat rectangles
// b) that the only way to have almost working transparency is a set of NURBS bodies.
// (Or may be other topology types like plane also work
// demonstration how non-transparent materials work the same for all kinds of objects  

  if (1) { // demonstration how non-transparent materials work the same for all kinds of objects  
    const size_t NUMBER_OF_PATCHES = 6;
    double vertices[NUMBER_OF_PATCHES][4][3] = 
    {
      { // Patch 0
       {-1,-1,-1},
       { 1,-1,-1},
       {-1, 1,-1},
       { 1, 1,-1}
      },
      { // Patch 1
       {-1,-1, 1},
       { 1,-1, 1},
       {-1, 1, 1},
       { 1, 1, 1}
      },
      { // Patch 2
       {-1,-1,-1},
       { 1,-1,-1},
       {-1,-1, 1},
       { 1,-1, 1}
      },
      { // Patch 3
       {-1, 1,-1},
       { 1, 1,-1},
       {-1, 1, 1},
       { 1, 1, 1}
      },
      { // Patch 4
       {-1,-1,-1},
       {-1, 1,-1},
       {-1,-1, 1},
       {-1, 1, 1}
      },
      { // Patch 5
       { 1,-1,-1},
       { 1, 1,-1},
       { 1,-1, 1},
       { 1, 1, 1}
      }
    };
    const size_t NUMBER_OF_BOXES = 6;
    double shifted_vertices[NUMBER_OF_BOXES][NUMBER_OF_PATCHES][4][3];
    for(size_t box = 0; box < NUMBER_OF_BOXES; ++box)
      for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
        for(size_t i = 0; i < 4; ++i)
        {
          shifted_vertices[box][patch][i][0] = vertices[patch][i][0]+3*box;
          shifted_vertices[box][patch][i][1] = vertices[patch][i][1];
          shifted_vertices[box][patch][i][2] = vertices[patch][i][2]-2;
        }
    PRCmaterial materialGreen(
      RGBAColour(0.0,0.18,0.0),
      RGBAColour(0.0,0.878431,0.0),
      RGBAColour(0.0,0.32,0.0),
      RGBAColour(0.0,0.072,0.0),
      1.0,0.1);

    file.begingroup("TransparentBox");
    file.begingroup("SetOfNURBSBodies");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[0][patch], materials[(patch*5)%N_COLOURS], 0);
    }
    file.endgroup();
    file.begingroup("NURBSFacesPRCNOBREAK");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[1][patch], materials[(patch*5)%N_COLOURS], 0);
    }
    file.endgroup();
    file.begingroup("TessellatedPRCTESS");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[2][patch], materials[(patch*5)%N_COLOURS], 0);
    }
    file.endgroup();
    file.endgroup();

    file.begingroup("Box");
    file.begingroup("TessellatedPRCTESS");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[3][patch], materialGreen, 0);
    }
    file.endgroup();
    file.begingroup("NURBSFaces");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[4][patch], materialGreen, 0);
    }
    file.endgroup();
    file.begingroup("SetOfNURBSBodiesPRCDOBREAK");
    for(size_t patch = 0; patch < NUMBER_OF_PATCHES; ++patch)
    {
      file.addRectangle(shifted_vertices[5][patch], materialGreen, 0);
    }
    file.endgroup();
    file.endgroup();
  }

  if(0) { // test disk
    PRCTopoContext *diskContext = new PRCTopoContext;
    diskContext->granularity=0.0; // zero gives best quality
    uint32_t context_index = file.addTopoContext(diskContext);
   
    PRCBrepData *body = new PRCBrepData;
    uint32_t body_index = diskContext->addBrepData(body);
    PRCConnex *connex = new PRCConnex;
    body->addConnex(connex);
    PRCShell *shell = new PRCShell;
    // shell->shell_is_closed = true;
    connex->addShell(shell);
    PRCFace *face = new PRCFace;
    shell->addFace(face,2);
    PRCRuled *surface = new PRCRuled;
    face->setSurface(surface);
        
    PRCCircle *first_curve = new PRCCircle;
    first_curve->radius = 1;
    surface->setFirstCurve(first_curve);
    PRCCircle *second_curve = new PRCCircle;
    second_curve->radius = 0;
    surface->setSecondCurve(second_curve);

    surface->uv_domain.min.x = 0;
    surface->uv_domain.max.x = 1;
    surface->uv_domain.min.y = 0;
    surface->uv_domain.max.y = 2*M_PI;
    surface->parameterization_on_v_coeff_a = -1;
    surface->parameterization_on_v_coeff_b = 2*M_PI;

    surface->has_transformation = true;
    surface->geometry_is_2D = false;
    surface->behaviour = PRC_TRANSFORMATION_Translate|PRC_TRANSFORMATION_Rotate;
    surface->origin.Set(0,0,0);
    surface->x_axis.Set(1,0,0);
    surface->y_axis.Set(0,1,0);

    PRCBrepModel *brepmodel = new PRCBrepModel("disk");
    brepmodel->context_id = context_index;
    brepmodel->body_id = body_index;
    brepmodel->is_closed = true; // we do not need to see the tube from inside
    brepmodel->index_of_line_style = 0;
    file.addBrepModel(brepmodel);

    PRCSingleWireBody *firstCurveBody = new PRCSingleWireBody;
    uint32_t first_curve_body_index = diskContext->addSingleWireBody(firstCurveBody);
    PRCWireEdge *firstCurveEdge = new PRCWireEdge;
    firstCurveBody->setWireEdge(firstCurveEdge);
    firstCurveEdge->curve_3d = surface->first_curve;
    PRCWire *firstCurveWire = new PRCWire("firstCurveWire");
    firstCurveWire->index_of_line_style = 0;
    firstCurveWire->context_id = context_index;
    firstCurveWire->body_id = first_curve_body_index;
    file.addWire(firstCurveWire);
  } 

  if(1) {
    PRCmaterial materialGreen(
      RGBAColour(0.0,0.18,0.0),
      RGBAColour(0.0,0.878431,0.0),
      RGBAColour(0.0,0.32,0.0),
      RGBAColour(0.0,0.072,0.0),
      1.0,0.1);

     const double disk_origin[3] = {11,0,2};
     const double disk_x_axis[3] = {1,0,0};
     const double disk_y_axis[3] = {0,-1,0};
     const double disk_scale = 2;
     file.begingroup("diskPRCCLOSED");
     file.addDisk(1,materialGreen,0.01,disk_origin,disk_x_axis,disk_y_axis,disk_scale);
     file.endgroup();
     const double hs_origin[3] = {11,0,2};
     const double hs_x_axis[3] = {1,0,0};
     const double hs_y_axis[3] = {0,1,0};
     const double hs_scale = 2;
     file.begingroup("hemispherePRCCLOSED");
     file.addHemisphere(1,materialGreen,0.01,hs_origin,hs_x_axis,hs_y_axis,hs_scale);
     file.endgroup();
     const double cyl_origin[3] = {11,0,1};
     const double cyl_x_axis[3] = {1,0,0};
     const double cyl_y_axis[3] = {0,1,0};
     const double cyl_scale = 1;
     file.begingroup("cylinderPRCCLOSED");
     file.addCylinder(1,1,materialGreen,0.01,cyl_origin,cyl_x_axis,cyl_y_axis,cyl_scale);
     file.endgroup();
     const double sp_origin[3] = {11,0,1};
     const double sp_x_axis[3] = {1,0,0};
     const double sp_y_axis[3] = {0,1,0};
     const double sp_scale = 1;
     file.begingroup("spherePRCCLOSED");
     file.addSphere(0.5,materialGreen,0.01,sp_origin,sp_x_axis,sp_y_axis,sp_scale);
     file.endgroup();
     const double tor_origin[3] = {11,0,0};
     const double tor_x_axis[3] = {1,0,0};
     const double tor_y_axis[3] = {0,1,0};
     const double tor_scale = 1;
     file.begingroup("torusPRCCLOSED");
     file.addTorus(0.5,0.1,0,360,materialGreen,0.01,tor_origin,tor_x_axis,tor_y_axis,tor_scale);
     file.endgroup();
  }

  if(0) { // Blend01 tube around a Composite curve - no good at corners
  const size_t NUMBER_OF_POINTS = 4;
  double points[NUMBER_OF_POINTS][3];
  points[0][0] = 1; points[0][1] = 0;                 points[0][2] = 0;
  points[1][0] = 1; points[1][1] = 0.552284749830793; points[1][2] = 0;
  points[2][0] = 0.552284749830793; points[2][1] = 1; points[2][2] = 0;
  points[3][0] = 0;                 points[3][1] = 1; points[3][2] = 0;
  double qoints[NUMBER_OF_POINTS][3];
  qoints[0][0] = 0; qoints[0][1] = 1;                 qoints[0][2] = 0;
  qoints[1][0] = 0; qoints[1][1] = 1; qoints[1][2] = 0.552284749830793;
  qoints[2][0] = 0; qoints[2][1] = 0.552284749830793; qoints[2][2] = 1;
  qoints[3][0] = 0;                 qoints[3][1] = 0; qoints[3][2] = 1;
//  for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
//  {
//    points[i][0] = 3.5*cos(3.0*i/NUMBER_OF_POINTS*2.0*M_PI);
//    points[i][1] = 3.5*sin(3.0*i/NUMBER_OF_POINTS*2.0*M_PI);
//    points[i][2] = 5.0*i/NUMBER_OF_POINTS-1.0;
//  }

    double knots[3+NUMBER_OF_POINTS+1];
    knots[0] = 1;
    for(size_t i = 1; i < 3+NUMBER_OF_POINTS; ++i)
    {
      knots[i] = (i+2)/3; // integer division is intentional
    }
    knots[3+NUMBER_OF_POINTS] = (3+NUMBER_OF_POINTS+1)/3;
   
    PRCTopoContext *tubeContext = new PRCTopoContext;
    tubeContext->granularity=0.0;  // zero gives best quality
    uint32_t context_index = file.addTopoContext(tubeContext);
   
    PRCBrepData *body = new PRCBrepData;
    uint32_t body_index = tubeContext->addBrepData(body);
    PRCConnex *connex = new PRCConnex;
    body->addConnex(connex);
    PRCShell *shell = new PRCShell;
    connex->addShell(shell);
    PRCFace *face = new PRCFace;
    shell->addFace(face);
    PRCBlend01 *surface = new PRCBlend01;
    face->setSurface(surface);
        
    PRCNURBSCurve *center_curve = new PRCNURBSCurve;
    center_curve->is_rational = false;
    center_curve->degree = 3;
    for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
      center_curve->control_point.push_back(PRCControlPoint(points[i][0]+0.0,points[i][1],points[i][2]));
    for(size_t i = 0; i < 3+NUMBER_OF_POINTS+1; ++i)
      center_curve->knot.push_back(knots[i]);
    surface->setCenterCurve(center_curve);
   
    PRCNURBSCurve *origin_curve = new PRCNURBSCurve;
    origin_curve->is_rational = false;
    origin_curve->degree = 3;
    for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
      origin_curve->control_point.push_back(PRCControlPoint(points[i][0]*1.01+0.0,points[i][1]*1.01,points[i][2]));
    for(size_t i = 0; i < 3+NUMBER_OF_POINTS+1; ++i)
      origin_curve->knot.push_back(knots[i]);
    surface->setOriginCurve(origin_curve);
   
    surface->uv_domain.min.x = 0;
    surface->uv_domain.max.x = 2*M_PI;
    surface->uv_domain.min.y = 1; // first knot
    surface->uv_domain.max.y = 2; // last knot
    PRCBrepModel *brepmodel = new PRCBrepModel("Tube");
    brepmodel->context_id = context_index;
    brepmodel->body_id = body_index;
    brepmodel->is_closed = true; // we do not need to see the tube from inside
    brepmodel->index_of_line_style = 0;
    file.addBrepModel(brepmodel);
   
    PRCSingleWireBody *originCurveBody = new PRCSingleWireBody;
    uint32_t origin_curve_body_index = tubeContext->addSingleWireBody(originCurveBody);
    PRCWireEdge *originCurveEdge = new PRCWireEdge;
    originCurveBody->setWireEdge(originCurveEdge);
    originCurveEdge->curve_3d = surface->origin_curve;
    PRCWire *originCurveWire = new PRCWire("originCurveWire");
    originCurveWire->index_of_line_style = 0;
    originCurveWire->context_id = context_index;
    originCurveWire->body_id = origin_curve_body_index;
    file.addWire(originCurveWire);
   
    PRCSingleWireBody *centerCurveBody = new PRCSingleWireBody;
    uint32_t center_curve_body_index = tubeContext->addSingleWireBody(centerCurveBody);
    PRCWireEdge *centerCurveEdge = new PRCWireEdge;
    centerCurveBody->setWireEdge(centerCurveEdge);
    centerCurveEdge->curve_3d = surface->center_curve;
    PRCWire *centerCurveWire = new PRCWire("centerCurveWire");
    centerCurveWire->index_of_line_style = 0;
    centerCurveWire->context_id = context_index;
    centerCurveWire->body_id = center_curve_body_index;
    file.addWire(centerCurveWire);

    PRCNURBSCurve *Center_curve = new PRCNURBSCurve;
    Center_curve->is_rational = false;
    Center_curve->degree = 3;
    for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
      Center_curve->control_point.push_back(PRCControlPoint(qoints[i][0],qoints[i][1],qoints[i][2]));
    for(size_t i = 0; i < 3+NUMBER_OF_POINTS+1; ++i)
      Center_curve->knot.push_back(knots[i]);
   
    PRCNURBSCurve *Origin_curve = new PRCNURBSCurve;
    Origin_curve->is_rational = false;
    Origin_curve->degree = 3;
    for(size_t i = 0; i < NUMBER_OF_POINTS; ++i)
      Origin_curve->control_point.push_back(PRCControlPoint(qoints[i][0],qoints[i][1]*1.01,qoints[i][2]*1.01));
    for(size_t i = 0; i < 3+NUMBER_OF_POINTS+1; ++i)
      Origin_curve->knot.push_back(knots[i]);
   
    PRCSingleWireBody *OriginCurveBody = new PRCSingleWireBody;
    uint32_t Origin_curve_body_index = tubeContext->addSingleWireBody(OriginCurveBody);
    PRCWireEdge *OriginCurveEdge = new PRCWireEdge;
    OriginCurveBody->setWireEdge(OriginCurveEdge);
    OriginCurveEdge->setCurve(Origin_curve);
    PRCWire *OriginCurveWire = new PRCWire("OriginCurveWire");
    OriginCurveWire->index_of_line_style = 0;
    OriginCurveWire->context_id = context_index;
    OriginCurveWire->body_id = Origin_curve_body_index;
    file.addWire(OriginCurveWire);
   
    PRCSingleWireBody *CenterCurveBody = new PRCSingleWireBody;
    uint32_t Center_curve_body_index = tubeContext->addSingleWireBody(CenterCurveBody);
    PRCWireEdge *CenterCurveEdge = new PRCWireEdge;
    CenterCurveBody->setWireEdge(CenterCurveEdge);
    CenterCurveEdge->setCurve(Center_curve);
    PRCWire *CenterCurveWire = new PRCWire("CenterCurveWire");
    CenterCurveWire->index_of_line_style = 0;
    CenterCurveWire->context_id = context_index;
    CenterCurveWire->body_id = Center_curve_body_index;
    file.addWire(CenterCurveWire);

    PRCComposite *compositeCenter_curve = new PRCComposite;
    compositeCenter_curve->base_curve.push_back(centerCurveEdge->curve_3d);
    compositeCenter_curve->base_sense.push_back(true);
    compositeCenter_curve->base_curve.push_back(CenterCurveEdge->curve_3d);
    compositeCenter_curve->base_sense.push_back(true);
    compositeCenter_curve->is_closed = false;
    compositeCenter_curve->interval.min = 0;
    compositeCenter_curve->interval.max = 2;

    PRCSingleWireBody *compositeCenterCurveBody = new PRCSingleWireBody;
    uint32_t compositeCenter_curve_body_index = tubeContext->addSingleWireBody(compositeCenterCurveBody);
    PRCWireEdge *compositeCenterCurveEdge = new PRCWireEdge;
    compositeCenterCurveBody->setWireEdge(compositeCenterCurveEdge);
    compositeCenterCurveEdge->setCurve(compositeCenter_curve);
    PRCWire *compositeCenterCurveWire = new PRCWire("compositeCenterCurveWire");
    compositeCenterCurveWire->index_of_line_style = 0;
    compositeCenterCurveWire->context_id = context_index;
    compositeCenterCurveWire->body_id = compositeCenter_curve_body_index;
    file.addWire(compositeCenterCurveWire);
    
    PRCComposite *compositeOrigin_curve = new PRCComposite;
    compositeOrigin_curve->base_curve.push_back(originCurveEdge->curve_3d);
    compositeOrigin_curve->base_sense.push_back(true);
    compositeOrigin_curve->base_curve.push_back(OriginCurveEdge->curve_3d);
    compositeOrigin_curve->base_sense.push_back(true);
    compositeOrigin_curve->is_closed = false;
    compositeOrigin_curve->interval.min = 0;
    compositeOrigin_curve->interval.max = 2;

    PRCSingleWireBody *compositeOriginCurveBody = new PRCSingleWireBody;
    uint32_t compositeOrigin_curve_body_index = tubeContext->addSingleWireBody(compositeOriginCurveBody);
    PRCWireEdge *compositeOriginCurveEdge = new PRCWireEdge;
    compositeOriginCurveBody->setWireEdge(compositeOriginCurveEdge);
    compositeOriginCurveEdge->setCurve(compositeOrigin_curve);
    PRCWire *compositeOriginCurveWire = new PRCWire("compositeOriginCurveWire");
    compositeOriginCurveWire->index_of_line_style = 0;
    compositeOriginCurveWire->context_id = context_index;
    compositeOriginCurveWire->body_id = compositeOrigin_curve_body_index;
    file.addWire(compositeOriginCurveWire);
    
    PRCBrepData *cbody = new PRCBrepData;
    uint32_t cbody_index = tubeContext->addBrepData(cbody);
    PRCConnex *cconnex = new PRCConnex;
    cbody->addConnex(cconnex);
    PRCShell *cshell = new PRCShell;
    cconnex->addShell(cshell);
    PRCFace *cface = new PRCFace;
    cshell->addFace(cface);
    PRCBlend01 *csurface = new PRCBlend01;
    cface->setSurface(csurface);

    csurface->uv_domain.min.x = 0;
    csurface->uv_domain.max.x = 2*M_PI;
    csurface->uv_domain.min.y = 0; // first knot
    csurface->uv_domain.max.y = 2; // last knot
    csurface->center_curve = compositeCenterCurveEdge->curve_3d;
    csurface->origin_curve = compositeOriginCurveEdge->curve_3d;
    PRCBrepModel *cbrepmodel = new PRCBrepModel("cTube");
    cbrepmodel->context_id = context_index;
    cbrepmodel->body_id = cbody_index;
    cbrepmodel->is_closed = true; // we do not need to see the tube from inside
    cbrepmodel->index_of_line_style = 0;
    file.addBrepModel(cbrepmodel);
   
        
  } 

 file.finish();

  return 0;
}

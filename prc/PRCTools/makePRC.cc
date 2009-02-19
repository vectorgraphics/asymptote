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
#include <unistd.h>
#include <cstdlib>
#include "../oPRCFile.h"

using namespace std;

void readPoints(istream &is, unsigned int n, double p[][3])
{
  for(unsigned int i = 0; i < n; ++i)
  {
    is >> p[i][0] >> p[i][1] >> p[i][2];
  }
  if(!is)
  {
    cerr << "Error reading list of points." << endl;
    exit(1);
  }
}

void readDoubles(istream &is, unsigned int n, double d[])
{
  for(unsigned int i = 0; i < n; ++i)
  {
    is >> d[i];
  }
  if(!is)
  {
    cerr << "Error reading list of doubles." << endl;
    exit(1);
  }
}

int main(int argc, char **argv)
{
  const char *oFileName = "output.prc";

  int c;
  opterr = 0;
  while ((c = getopt (argc, argv, "o:")) != -1)
    switch (c)
    {
      case 'o':
        oFileName = optarg;
        break;
      case '?':
        if (optopt == 'o')
          cerr << "Option '-o' requires an argument, the filename." << endl;
        else
          cerr << "Unrecognized option '-" << (char)optopt << "'." << endl;
        exit(1);
        break;
      default:
        exit(1);
    }
  istream *ins = NULL;
  if(optind < argc)
  {
    ins = new ifstream(argv[optind]);
    if(!*ins)
    {
      cerr << "Error opening input file " << argv[optind] << endl;
      exit(1);
    }
  }
  else
  {
    ins = &cin;
  }

  ofstream outf(oFileName);
  if(!outf)
  {
    cerr << "Error opening output file " << oFileName << endl;
    exit(1);
  }

  oPRCFile oPRC(outf);

  string entityType;
  while(*ins)
  {
    *ins >> entityType;
    if(!*ins) break;
    if(entityType == "Line")
    {
      double r,g,b,a;
      unsigned int numberOfPoints;
      *ins >> r >> g >> b >> a >> numberOfPoints;
      if(!*ins)
      {
        cerr << "Error reading line data." << endl;
        exit(1);
      }
      else
      {
        double (*points)[3] = new double[numberOfPoints][3];
        readPoints(*ins,numberOfPoints,points);
        oPRC.add(new PRCline(&oPRC,numberOfPoints,points,
                 *new RGBAColour(r,g,b,a)));
      }
    }
    else if(entityType == "Curve")
    {
      double r,g,b,a;
      unsigned int numberOfPoints;
      unsigned int degree;
      *ins >> r >> g >> b >> a >> degree >> numberOfPoints;
      if(!*ins)
      {
        cerr << "Error reading curve data." << endl;
        exit(1);
      }
      else
      {
        double (*points)[3] = new double[numberOfPoints][3];
        double *knots = new double[degree+numberOfPoints+1];
        readPoints(*ins,numberOfPoints,points);
        readDoubles(*ins,degree+numberOfPoints+1,knots);
        oPRC.add(new PRCcurve(&oPRC,degree,numberOfPoints,points,
                 knots,*new RGBAColour(r,g,b,a)));
      }
    }
    else if(entityType == "Surface")
    {
      double r,g,b,a;
      unsigned int numberOfPointsU,numberOfPointsV;
      unsigned int degreeU,degreeV;
      *ins >> r >> g >> b >> a >> degreeU >> degreeV >> numberOfPointsU
          >> numberOfPointsV;
      if(!*ins)
      {
        cerr << "Error reading surface data." << endl;
        exit(1);
      }
      else
      {
        double (*points)[3] = new double[numberOfPointsU*numberOfPointsV][3];
        double *knotsU = new double[degreeU+numberOfPointsU+1];
        double *knotsV = new double[degreeV+numberOfPointsV+1];
        readPoints(*ins,numberOfPointsU*numberOfPointsV,points);
        readDoubles(*ins,degreeU+numberOfPointsU+1,knotsU);
        readDoubles(*ins,degreeV+numberOfPointsV+1,knotsV);
        oPRC.add(new PRCsurface(&oPRC,degreeU,degreeV,numberOfPointsU,
                 numberOfPointsV,points,knotsU,knotsV,
                 *new RGBAColour(r,g,b,a)));
      }
    }
    else
    {
      cerr << "Unrecognized entity type " << entityType << endl;
      exit(1);
    }
  }

  if(ins && ins != &cin)
    delete ins;


  oPRC.finish();

  outf.close();

  return 0;
}

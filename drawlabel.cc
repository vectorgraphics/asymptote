/*****
 * drawlabel.cc
 * John Bowman 2003/04/07
 *
 * Add a label to a picture.
 *****/

#include <sstream>

#include "drawlabel.h"
#include "settings.h"
#include "util.h"

namespace camp {
  
extern string texready;
pen drawElement::lastpen;

void drawLabel::labelwarning(const char *action) 
{
  cerr << "warning: label \"" << label 
	       << "\" " << action << " to avoid overwriting" << endl;
}
  
bool drawLabel::texbounds(iopipestream& tex, string& s, bool warn)
{
  string texbuf;
  tex << "\\setbox\\ASYbox=\\hbox{" << stripblanklines(s) << "}\n\n";
  tex.wait(texready.c_str(),"! ");
  tex << "\\showthe\\wd\\ASYbox\n";
  tex >> texbuf;
  if(texbuf[0] == '>' && texbuf[1] == ' ')
    width=atof(texbuf.c_str()+2)*tex2ps;
  else {
    if(settings::getSetting<bool>("inlinetex")) {
      if(settings::getSetting<bool>("debug") && warn) {
	ostringstream buf;
	buf << "Cannot determine size of label \"" << s << "\"";
	reportWarning(buf);
      }
      tex << "\n";
      tex.wait("\n*","! ");
      return false;
    } else reportError("Can't read label width");
  }
  tex << "\n";
  tex.wait("\n*","! ");
  tex << "\\showthe\\ht\\ASYbox\n";
  tex >> texbuf;
  if(texbuf[0] == '>' && texbuf[1] == ' ')
    height=atof(texbuf.c_str()+2)*tex2ps;
  else reportError("Can't read label height");
  tex << "\n";
  tex.wait("\n*","! ");
  tex << "\\showthe\\dp\\ASYbox\n";
  tex >> texbuf;
  if(texbuf[0] == '>' && texbuf[1] == ' ')
    depth=atof(texbuf.c_str()+2)*tex2ps;
  else reportError("Can't read label depth");
  tex << "\n";
  tex.wait("\n*","! ");
     
  width *= fontscale*scale.getx();
  height *= fontscale*scale.gety();
  depth *= fontscale*scale.gety();
  return true;
}   

inline double urand()
{			  
  static const double factor=2.0/RAND_MAX;
  return rand()*factor-1.0;
}

void drawLabel::bounds(bbox& b, iopipestream& tex, boxvector& labelbounds,
		       bboxlist&)
{
  if(!settings::getSetting<bool>("tex")) {b += position; return;}
  pair rotation=expi(radians(angle));
  pen Pentype=pentype;
  double offset=0.3;
  static double fuzz=Pentype.size()/24.0;
  static double hfuzz=fuzz+offset;
  static double vfuzz=fuzz*scale.gety()+offset;
  
  if(!havebounds) {
    havebounds=true;
    if(Pentype.size() != lastpen.size() ||
       Pentype.Lineskip() != lastpen.Lineskip()) {
      tex <<  "\\fontsize{" << Pentype.size() << "}{" << Pentype.Lineskip()
	  << "}\\selectfont\n";
      tex.wait("\n*","! ");
    }
    
    string font=Pentype.Font();
    if(font != lastpen.Font()) {
      fontscale=1.0;
      string scaled=" scaled";
      size_t p=font.find(scaled);
      if(p < string::npos) {
	p += scaled.length();
	fontscale=atof(font.substr(p,string::npos).c_str())/1000.0;
      }
      tex <<  font << "\n";
      tex.wait("\n*","! ");
    }
    
    lastpen=Pentype;
    
    bool nullsize=size == "";
    if(!texbounds(tex,label,nullsize) && !nullsize)
      texbounds(tex,size,false);
    
    Align=align/rotation;
    double scale0=max(fabs(Align.getx()),fabs(Align.gety()));
    if(scale0) Align *= 0.5/scale0;
    Align -= pair(0.5,0.5);
    double Depth=(Pentype.Baseline() == NOBASEALIGN) ? depth : 0.0;
    texAlign=Align;
    if(Depth > 0) texAlign += pair(0.0,Depth/(height+Depth));
    Align.scale(width,height+Depth);
    Align += pair(0.0,Depth-depth);
    Align *= rotation;
  }

  // alignment point
  pair p=position+Align;
  double vertical=height+depth+vfuzz;
  pair A=p+pair(-hfuzz,-vfuzz)*rotation;
  pair B=p+pair(-hfuzz,vertical)*rotation;
  pair C=p+pair(width+hfuzz,vertical)*rotation;
  pair D=p+pair(width+hfuzz,-vfuzz)*rotation;
  
  if(pentype.Overwrite() != ALLOW && label != "") {
    size_t n=labelbounds.size();
    box Box=box(A,B,C,D);
    for(size_t i=0; i < n; i++) {
      if(labelbounds[i].intersect(Box)) {
	switch(pentype.Overwrite()) {
	case SUPPRESS:
	  labelwarning("suppressed");
	case SUPPRESSQUIET:
	  suppress=true; 
	  return;
	case MOVE:
	  labelwarning("moved");
	default:
	  break;
	}

	pair Align=(align == pair(0,0)) ? unit(pair(urand(),urand())) :
	  unit(align);
	double s=0.1*pentype.size();
	double dx=0, dy=0;
	if(Align.getx() > 0.1) dx=labelbounds[i].xmax()-Box.xmin()+s;
	if(Align.getx() < -0.1) dx=labelbounds[i].xmin()-Box.xmax()-s;
	if(Align.gety() > 0.1) dy=labelbounds[i].ymax()-Box.ymin()+s;
	if(Align.gety() < -0.1) dy=labelbounds[i].ymin()-Box.ymax()-s;
	pair offset=pair(dx,dy);
	position += offset;
	A += offset;
	B += offset;
	C += offset;
	D += offset;
	Box=box(A,B,C,D);
	i=0;
      }
    }
    labelbounds.resize(n+1);
    labelbounds[n]=Box;
  }
  
  Box=bbox();
  Box += A;
  Box += B;
  Box += C;
  Box += D;
  
  b += Box;
}

drawElement *drawLabel::transformed(const transform& t)
{
  static const pair origin=pair(0,0);
  pair offset=t*origin;
  return new drawLabel(label,size,
		       degrees((t*expi(radians(angle))-offset).angle()),
		       t*position,length(align)*unit(t*align-offset),scale,
		       pentype);
}

} //namespace camp

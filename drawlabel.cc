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

using std::list;
  
extern const char *texready;

namespace camp {

void drawLabel::labelwarning(const char *action) 
{
  cerr << "warning: label \"" << label 
	       << "\" " << action << " to avoid overwriting" << endl;
}
  
void drawLabel::bounds(bbox& b, iopipestream& tex,
		       std::vector<box>& labelbounds)
{
  if(!settings::texprocess) {b += position; return;}
  string texbuf;
  pair rotation=expi(radians(angle));
  pen Pentype=*pentype;
  static double fuzz=1.75;
  
  if(!(width || height || depth)) {
    tex <<  "\\fontsize{" << Pentype.size() << "}{" << Pentype.size()*1.2
	<< "}\\selectfont\n";
    tex.wait("\n*","! ");
    tex << "\\setbox\\ASYbox=\\hbox{" << stripblanklines(label) << "}\n\n";
    tex.wait(texready,"! ");
    tex << "\\showthe\\wd\\ASYbox\n";
    tex >> texbuf;
    if(texbuf[0] == '>' && texbuf[1] == ' ')
      width=atof(texbuf.c_str()+2)*tex2ps;
    else cerr << "Can't read label width\n";
    tex << "\n";
    tex.wait("\n*","! ");
    tex << "\\showthe\\ht\\ASYbox\n";
    tex >> texbuf;
    if(texbuf[0] == '>' && texbuf[1] == ' ')
      height=atof(texbuf.c_str()+2)*tex2ps;
    else cerr << "Can't read label height" << newl;
    tex << "\n";
    tex.wait("\n*","! ");
    tex << "\\showthe\\dp\\ASYbox\n";
    tex >> texbuf;
    if(texbuf[0] == '>' && texbuf[1] == ' ')
      depth=atof(texbuf.c_str()+2)*tex2ps;
    else cerr << "Can't read label depth" << newl;
    tex << "\n";
    tex.wait("\n*","! ");
     
    Align=align/rotation;
    double scale0=max(fabs(Align.getx()),fabs(Align.gety()));
    if(scale0) Align *= 0.5/scale0;
    Align -= pair(0.5,0.5);
    Align.scale(width,height+depth);
    Align += pair(0.0,depth);
    Align *= rotation;
  }

  // alignment point
  pair p=position+Align+pair(0,-depth)*rotation;
  pair A=p+pair(-fuzz,-fuzz)*rotation;
  pair B=p+pair(-fuzz,height+depth+fuzz)*rotation;
  pair C=p+pair(width+fuzz,height+depth+fuzz)*rotation;
  pair D=p+pair(width+fuzz,-fuzz)*rotation;
  
  if(settings::overwrite != 0) {
    size_t n=labelbounds.size();
    box Box=box(A,B,C,D);
    for(size_t i=0; i < n; i++) {
      if(labelbounds[i].intersect(Box)) {
	if(abs(settings::overwrite) == 1) {
	  suppress=true; 
	  if(settings::overwrite > 0) labelwarning("suppressed");
	  return;
	}

	pair Align=(align == pair(0,0)) ? pair(1,0) : unit(align);
	double s=1e-12;
	pair offset=-p;
	p=pair(Align.getx() > 0 ? 
	       max(p.getx(),(labelbounds[i].xmax()+s)) : 
	       Align.getx() == 0 ? p.getx() : min(p.getx(),
						  (labelbounds[i].xmin()-s)), 
	       Align.gety() > 0 ? 
	       max(p.gety(),(labelbounds[i].ymax()+s)) : 
	       Align.gety() == 0 ? p.gety() : min(p.gety(),
						  (labelbounds[i].ymin()-s))); 
	offset += p;
	position += offset;
	A += offset;
	B += offset;
	C += offset;
	D += offset;
	Box=box(A,B,C,D);
	if(settings::overwrite > 0) labelwarning("moved");
	i=0;
      }
    }
    labelbounds.resize(n+1);
    labelbounds[n]=Box;
  }
  
  b += A;
  b += B;
  b += C;
  b += D;
}

drawElement *drawLabel::transformed(const transform& t)
{
  static const pair origin=pair(0,0);
  pair offset=t*origin;
  return new drawLabel(label,
		       degrees((t*expi(radians(angle))-offset).angle()),
		       t*position,t*align-offset,pentype);
}

} //namespace camp

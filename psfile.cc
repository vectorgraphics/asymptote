/*****
 * psfile.cc
 * Andy Hammerlindl 2002/06/10
 *
 * Encapsulates the writing of commands to a PostScript file.
 * Allows identification and removal of redundant commands.
 *****/

#include <ctime>
#include <iomanip>

#include "psfile.h"
#include "settings.h"
#include "errormsg.h"

using std::string;
using std::ofstream;
using std::setw;
  
namespace camp {

psfile::psfile(const string& filename, const bbox& box, const pair& shift)
  : filename(filename), box(box), shift(shift), rawmode(false)
{
  if(filename != "") out=new ofstream(filename.c_str());
  else out=&std::cout;
  if(!out || !*out) {
    std::cerr << "Can't write to " << filename << std::endl;
    throw handled_error();
  }
}

psfile::~psfile()
{
  if(filename != "" && out) delete out;
}

void psfile::prologue()
{
  //*out << "%!PS" << newl;
  *out << "%!PS-Adobe-3.0 EPSF-3.0" << newl;
  BoundingBox(*out,box);
  *out << "%%Creator: " << settings::PROGRAM << " " << settings::VERSION
       <<  newl;

  time_t t; time(&t);
  struct tm *tt = localtime(&t);
  char prev = out->fill('0');
  *out << "%%CreationDate: " << tt->tm_year + 1900 << "."
       << setw(2) << tt->tm_mon+1 << "." << setw(2) << tt->tm_mday << " "
       << setw(2) << tt->tm_hour << ":" << setw(2) << tt->tm_min << ":"
       << setw(2) << tt->tm_sec << newl;
  out->fill(prev);

  *out << "%%Pages: 1" << newl;
  *out << "%%EndProlog" << newl;
  *out << "%%Page: 1 1" << newl;
}

void psfile::epilogue()
{
  *out << "showpage" << newl;
  *out << "%%EOF" << newl;
  out->flush();
}

void psfile::setpen(pen p)
{
  p.convert();
  if(p == lastpen) return;
    
  if(p.fillpattern() != "" && p.fillpattern() != lastpen.fillpattern()) 
    *out << p.fillpattern() << " setpattern" << newl;
  else if(p.cmyk() && (!lastpen.cmyk() ||
		       (p.cyan() != lastpen.cyan() || 
			p.magenta() != lastpen.magenta() || 
			p.yellow() != lastpen.yellow() ||
			p.black() != lastpen.black()))) {
    *out << p.cyan() << " " << p.magenta() << " " << p.yellow() << " " 
	 << p.black() << " setcmykcolor" << newl;
  } else if(p.rgb() && (!lastpen.rgb() || 
			(p.red() != lastpen.red() || 
			 p.green() != lastpen.green() || 
			 p.blue() != lastpen.blue()))) {
    *out << p.red() << " " << p.green() << " " << p.blue()
	 << " setrgbcolor" << newl;
  } else if(p.grayscale() && (!lastpen.grayscale() ||
			      p.gray() != lastpen.gray())) {
    *out << p.gray() << " setgray" << newl;
  }
    
  if(p.width() != lastpen.width()) {
    *out << " 0 " << p.width() << 
      " dtransform truncate idtransform setlinewidth pop" << newl;
  }
    
  if(p.cap() != lastpen.cap()) {
    *out << p.cap() << " setlinecap" << newl;
  }
    
  if(p.join() != lastpen.join()) {
    *out << p.join() << " setlinejoin" << newl;
  }
    
  if(p.stroke() != lastpen.stroke()) {
    *out << "[" << p.stroke() << "] 0 setdash" << newl;
  }
    
  lastpen=p;
}

void psfile::write(pen p)
{
  if(p.cmyk())
    *out << p.cyan() << " " << p.magenta() << " " << p.yellow() << " " 
	 << p.black();
  else if(p.rgb())
    *out << p.red() << " " << p.green() << " " << p.blue();
  else if(p.grayscale())
    *out << p.gray();
}
  
void psfile::write(path p, bool newPath)
{
  int n = p.size();
  assert(n != 0);

  if(newPath) newpath();

  if (n == 1) {
    moveto(p.point(0));
    rlineto(pair(0,0));
    stroke();
  }

  // Draw points
  moveto(p.point(0));
  for (int i = 1; i < n; i++) {
    if(p.straight(i-1)) lineto(p.point(i));
    else curveto(p.postcontrol(i-1), p.precontrol(i), p.point(i));
  }

  if (p.cyclic()) {
    if(p.straight(n-1)) lineto(p.point(0));
    else curveto(p.postcontrol(n-1), p.precontrol(0), p.point(0));
    closepath();
  }    
}

void psfile::shade(bool axial, const string& colorspace,
		   const pen& pena, const pair& a, double ra,
		   const pen& penb, const pair& b, double rb)
{
  *out << "<< /ShadingType " << (axial ? "2" : "3") << newl
       << "/ColorSpace /Device" << colorspace << newl
       << "/Coords [";
  write(a);
  if(!axial) write(ra);
  write(b);
  if(!axial) write(rb);
  *out << "]" << newl
       << "/Extend [true true]" << newl
       << "/Function" << newl
       << "<< /FunctionType 2" << newl
       << "/Domain [0 1]" << newl
       << "/C0 [";
  write(pena);
  *out << "]" << newl
       << "/C1 [";
  write(penb);
  *out << "]" << newl
       << "/N 1" << newl
       << ">>" << newl
       << ">>" << newl
       << "shfill" << newl;
}
  
} //namespace camp

/*****
 * psfile.cc
 * Andy Hammerlindl 2002/06/10
 *
 * Encapsulates the writing of commands to a PostScript file.
 * Allows identification and removal of redundant commands.
 *****/

#include <ctime>
#include <iomanip>
#include <sstream>

#include "psfile.h"
#include "settings.h"
#include "errormsg.h"
#include "array.h"

using std::ofstream;
using std::setw;
using vm::array;
using vm::read;
  
namespace camp {

void checkColorSpace(ColorSpace colorspace)
{
  switch(colorspace) {
  case DEFCOLOR:
  case INVISIBLE:
    reportError("Cannot shade with invisible pen");
  case PATTERN:
    reportError("Cannot shade with pattern");
    break;
  default:
    break;
  }
}
    
psfile::psfile(const string& filename, bool pdfformat)
  : filename(filename), pdfformat(pdfformat)
{
  if(filename.empty()) out=&std::cout;
  else out=new ofstream(filename.c_str());
  if(!out || !*out) {
    std::cerr << "Cannot write to " << filename << std::endl;
    throw handled_error();
  }
}

psfile::~psfile()
{
  if(out) {
    out->flush();
    if(!filename.empty()) {
      if(!out->good()) {
	std::ostringstream msg;
	msg << "Cannot write to " << filename;
	reportError(msg);
      }
      delete out;
    }
  }
}

void psfile::prologue(const bbox& box)
{
  *out << "%!PS-Adobe-3.0 EPSF-3.0" << newl;
  BoundingBox(box);
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
}

void psfile::setpen(pen p)
{
  p.convert();
  if(p == lastpen) return;
    
  if(pdfformat) {
    if(p.blend() != lastpen.blend()) 
      *out << "/" << p.blend() << " .setblendmode" << newl;
  
    if(p.opacity() != lastpen.opacity()) 
      *out << p.opacity() << " .setopacityalpha" << newl;
  }
  
  if(!p.fillpattern().empty() && p.fillpattern() != lastpen.fillpattern()) 
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
    *out << "[" << p.stroke() << "] " << std::setprecision(6) 
	 << p.linetype().offset << " setdash" << newl;
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

// Lattice shading
void psfile::shade(array *a, const bbox& b)
{
  size_t n=a->size();
  if(n == 0) return;
  
  array *a0=read<array *>(a,0);
  size_t m=a0->size();

  pen *p=read<pen *>(a0,0);
  p->convert();
  ColorSpace colorspace=p->colorspace();
  checkColorSpace(colorspace);
  
  unsigned ncomponents=ColorComponents[colorspace];
  
  *out << "<< /ShadingType 1" << newl
       << "/Matrix ";

  write(matrix(b.Min(),b.Max()));
  *out << newl;
  *out << "/ColorSpace /Device" << ColorDeviceSuffix[colorspace] << newl
       << "/Function" << newl
       << "<< /FunctionType 0" << newl
       << "/Order 1" << newl
       << "/Domain [0 1 0 1]" << newl
       << "/Range [0 1 0 1 0 1]" << newl
       << "/Decode [";
  
  for(unsigned i=0; i < ncomponents; ++i)
    *out << "0 1 ";
  
  *out << "]" << newl;
  *out << "/BitsPerSample 8" << newl;
  *out << "/Size [" << m << " " << n << "]" << newl
       << "/DataSource <" << newl;
  for(size_t i=n; i > 0;) {
    array *ai=read<array *>(a,--i);
    checkArray(ai);
    size_t aisize=ai->size();
    if(aisize != m) reportError("shading matrix must be rectangular");
    for(size_t j=0; j < m; j++) {
	pen *p=read<pen *>(ai,j);
	p->convert();
	if(p->colorspace() != colorspace)
	  reportError("inconsistent shading colorspaces");
	writeHex(p,ncomponents);
      }
    }
  *out << ">" << newl
       << ">>" << newl
       << ">>" << newl
       << "shfill" << newl;
}

// Axial and radial shading
void psfile::shade(bool axial, const ColorSpace &colorspace,
		   const pen& pena, const pair& a, double ra,
		   const pen& penb, const pair& b, double rb)
{
  checkColorSpace(colorspace);
  
  *out << "<< /ShadingType " << (axial ? "2" : "3") << newl
       << "/ColorSpace /Device" << ColorDeviceSuffix[colorspace] << newl
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
  
// Gouraud shading
void psfile::shade(array *pens, array *vertices, array *edges)
{
  size_t size=pens->size();
  if(size == 0) return;
  
  pen *p=read<pen *>(pens,0);
  p->convert();
  ColorSpace colorspace=p->colorspace();
  checkColorSpace(colorspace);

  *out << "<< /ShadingType 4" << newl
       << "/ColorSpace /Device" << ColorDeviceSuffix[colorspace] << newl
       << "/DataSource [" << newl;
  for(size_t i=0; i < size; i++) {
    write(read<int>(edges,i));
    write(read<pair>(vertices,i));
    pen *p=read<pen *>(pens,i);
    p->convert();
    if(p->colorspace() != colorspace)
      reportError("inconsistent shading colorspaces");
    *out << " ";
    write(*p);
    *out << newl;
  }
  *out << "]" << newl
       << ">>" << newl
       << "shfill" << newl;
}
 
inline unsigned int byte(double r) // Map [0,1] to [0,255]
{
  if(r < 0.0) r=0.0;
  else if(r > 1.0) r=1.0;
  int a=(int)(256.0*r);
  if(a == 256) a=255;
  return a;
}

void psfile::writeHex(pen *p, int ncomponents) {
  switch(ncomponents) {
  case 0:
    break;
  case 1: 
    writeHex(byte(p->gray())); 
    *out << newl;
    break;
  case 3:
    writeHex(byte(p->red())); 
    writeHex(byte(p->green())); 
    writeHex(byte(p->blue())); 
    *out << newl;
    break;
  case 4:
    writeHex(byte(p->cyan())); 
    writeHex(byte(p->magenta())); 
    writeHex(byte(p->yellow())); 
    writeHex(byte(p->black())); 
    *out << newl;
  default:
    break;
  }
}

void psfile::image(array *a, array *P)
{
  size_t asize=a->size();
  size_t Psize=P->size();
  
  if(asize == 0 || Psize == 0) return;
  
  array *a0=read<array *>(a,0);
  size_t a0size=a0->size();
  
  pen *p=read<pen *>(P,0);
  p->convert();
  ColorSpace colorspace=p->colorspace();
  checkColorSpace(colorspace);
  unsigned ncomponents=ColorComponents[colorspace];
  
  *out << "/Device" << ColorDeviceSuffix[colorspace] << " setcolorspace" 
       << newl
       << "<<" << newl
       << "/ImageType 1" << newl
       << "/Width " << a0size << newl
       << "/Height " << asize << newl
       << "/BitsPerComponent 8" << newl
       << "/Decode [";
  
  for(unsigned i=0; i < ncomponents; ++i)
    *out << "0 1 ";
  
  *out << "]" << newl
       << "/ImageMatrix [" << a0size << " 0 0 " << asize << " 0 0]" << newl
       << "/DataSource currentfile /ASCIIHexDecode filter" << newl
       << ">>" << newl
       << "image" << newl;
  
  double min=read<double>(a0,0);
  double max=min;
  for(size_t i=0; i < asize; i++) {
    array *ai=read<array *>(a,i);
    for(size_t j=0; j < a0size; j++) {
	double val=read<double>(ai,j);
	if(val > max) max=val;
	else if(val < min) min=val;
    }
  }
  
  double step=(max == min) ? 0.0 : (Psize-1)/(max-min);
  
  for(size_t i=0; i < asize; i++) {
    array *ai=read<array *>(a,i);
    for(size_t j=0; j < a0size; j++) {
      double val=read<double>(ai,j);
      pen *p=read<pen *>(P,(int) ((val-min)*step+0.5));
      p->convert();

      if(p->colorspace() != colorspace)
	reportError("inconsistent colorspaces in palette");
  
      writeHex(p,ncomponents);
    }
  }
  
  *out << ">" << endl;
}

} //namespace camp

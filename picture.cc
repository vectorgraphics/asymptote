/*****
 * picture.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a picture as a list of drawElements and handles its output to 
 * PostScript. 
 *****/

//#include <errno.h>

#include "errormsg.h"
#include "picture.h"
#include "util.h"
#include "settings.h"
#include "interact.h"

using std::list;
using std::ifstream;
using std::ofstream;

using namespace settings;

iopipestream tex; // Bi-directional pipe to latex (to find label bbox)

namespace camp {

static double pdfoffset=2.0;
  
picture::~picture()
{
}

void picture::prepend(drawElement *p)
{
  assert(p);

  nodes.push_front(p);
}

void picture::append(drawElement *p)
{
  assert(p);

  nodes.push_back(p);
}

void picture::add(picture &pic)
{
  if (&pic == this) return;

  // STL's funny way of copying one list into another.
  copy(pic.nodes.begin(), pic.nodes.end(), inserter(nodes, nodes.end()));
}

bbox picture::bounds()
{
  list<drawElement*>::iterator p;
  // Check to see if there are any labels yet
  if(!labels && settings::texprocess) {
    for (p = nodes.begin(); p != nodes.end(); ++p) {
      assert(*p);
      if((*p)->islabel())
	labels=true;
    }
  }
  
  if(labels) texinit();
  
  std::vector<box> labelbounds;  

  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->bounds(b,tex,labelbounds);
  }

  return b;
}

void picture::texinit() {
  if(TeXinitialized) return;
  
  tex.open("latex");
  texdocumentclass(tex);
  
  texpreamble(tex);

  tex << "\n";
  tex.wait("(Please type a command or say `\\end')\n*","! ");

  TeXinitialized=true;
}
  
bool picture::texoutput(const string& psname, const string& epsname,
			const string& prefix) 
{
  bool status = true;
  string texname=auxname(prefix,"tex");
  
  texfile texfile(texname,psname,b);
    
  list<drawElement*>::iterator p;
  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if (!(*p)->setup(&texfile))
      status = false;
  }
  
  texfile.prologue();
    
  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if (!(*p)->write(&texfile))
      status = false;
  }
  texfile.epilogue();
    
  if(status)
    status=texprocess(texname,epsname,prefix);
  
  if(!keep) unlink(psname.c_str());
    
  return status;
}

bool picture::texprocess(const string& texname, const string& outname,
			 const string& prefix)
{
  int status=0;
  std::ifstream outfile;
  
  outfile.open(texname.c_str());
  if(outfile) {
    outfile.close();
    ostringstream cmd;
    cmd << "latex \\scrollmode\\input " << texname;
    status=System(cmd,verbose <= 2);
    if(status) return false;
    string dviname=auxname(prefix,"dvi");
    outfile.open(dviname.c_str());
    if(!outfile) {
      if(verbose <= 2) System(cmd);
      return false;
    }
    outfile.close();
    
    ostringstream dcmd;
    double width=b.right-b.left;
    double height=b.top-b.bottom;
    // Magic dvips offsets:
    double hoffset=-128.2;
    double voffset=(height < 10.6) ? -136.1+height : -125.5;

    if(outputformat == "pdf") {
      voffset += pdfoffset;
      height += pdfoffset;
    }

    hoffset += printerOffset.getx();
    width += printerOffset.getx();
    height += printerOffset.gety();
    
    dcmd << "dvips -O " << hoffset << "bp," << voffset << "bp -T "
	 <<  width+2 << "bp," << height+2 << "bp"; 
    if(verbose <= 2) dcmd << " -q";
    dcmd << " -o " << outname << " " << dviname;
    status=System(dcmd);

    if(!keep) { // Delete temporary files.
      unlink("texput.log");
      unlink(texname.c_str());
      unlink(dviname.c_str());
      unlink(auxname(prefix,"aux").c_str());
      unlink(auxname(prefix,"log").c_str());
    }
  }
  if(status) return false;
  return true;
}

bool picture::postprocess(const string& epsname, const string& outname,
			  bool wait)
{
  int status=0;
  ostringstream cmd;
  bool epsformat=(outputformat == "" || outputformat == "eps");
  bool pdfformat=(outputformat == "pdf");
  bool tgifformat=(outputformat == "tgif");
  
  if(!epsformat) {
    if(pdfformat) cmd << "gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop"
		      << " -sOutputFile=" << outname << " " << epsname;
    else {
      double res=deconstruct*72;
      cmd << "convert";
      if(tgifformat) cmd << " -density " << res << "x" << res;
      cmd << " eps:" << epsname;
      if(tgifformat) cmd << " -transparent white gif";
      else cmd << " " << outputformat;  
      cmd << ":" << outname;
    }
    System(cmd);
    if(!keep) unlink(epsname.c_str());
  }
  
  if(verbose > (tgifformat ? 1 : 0)) cout << "Wrote " << outname << endl;
  static bool first=true;
  static int pid;
  if(view && !deconstruct) {
    ostringstream cmd;
    if(epsformat || pdfformat) {
      if (first || !interact::virtualEOF) {
	first=false;
	cmd << "gv ";
	if(!wait) cmd << "-watch ";
        cmd << outname;
	status=System(cmd,false,wait,&pid);
	if(status) return false;
      } else {
//	kill(pid,SIGHUP);
      }
    } else {
	cmd << "display " << outname;
	status=System(cmd,false,wait);
	if(status) return false;
    }
  }
  
  return true;
}

bool picture::shipout(const string& prefix, const string& format, bool wait)
{
  if(interact::interactive && (suppressOutput || upToDate)) return true;
  upToDate=true;
  
  checkFormatString(format);
  outputformat=(format == "" ? outformat : format);
  bool tgifformat=(outputformat == "tgif");
  
  static std::ofstream bboxout;
  if(deconstruct && !tgifformat) {
    bboxout.close();
    if(view) {
      ostringstream cmd;
      cmd << "xasy " << buildname(prefix) 
	  << " " << ShipoutNumber << " " << buildname(settings::outname);
      System(cmd,false,true);
    }
    ShipoutNumber++;
    return true;
  }
      
  bool epsformat=(outputformat == "" || outputformat == "eps");
  string outname=tgifformat ? "."+buildname(prefix,"gif") :
    buildname(prefix,outputformat);
  string epsname=epsformat ? outname : auxname(prefix,"eps");
  
  bounds();
  
  string psname=labels ? auxname(prefix,"ps") : epsname;
  bbox bpos=b;
  
  if(deconstruct) {
      if(!bboxout.is_open()) {
	bboxout.open(("."+buildname(prefix,"box")).c_str());	
	bboxout << deconstruct << endl;
      }
      bbox bscaled=b;
      bscaled *= deconstruct;
      bboxout << bscaled << endl;
  } else {
  // Avoid negative bounding box coordinates; adjust for printer Offset
    bboxshift=pair(-min(b.left,-printerOffset.getx()),
		   -min(b.bottom,-printerOffset.gety()));
    bpos.shift(bboxshift);
  }
  
  if(bpos.right <= bpos.left && bpos.top <= bpos.bottom) { // null picture
    unlink(outname.c_str());
    return true;
  }
  
  if(!labels && outputformat == "pdf") {
    bpos.right += pdfoffset;
    bpos.top += pdfoffset;
  }
  
  psfile out(psname,bpos,bboxshift);
  out.prologue();
  bool status = true;
  
  list<drawElement*>::iterator p;
  for(p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if(!(*p)->draw(&out))
      status = false;
  }

  out.epilogue();
  
  if(status && labels)
    status=texoutput(psname,epsname,prefix);
  
  if(status)
    status=postprocess(epsname,outname,wait);
  
  if(!tgifformat) outnameStack->push_back(outname);
  
  return status;
}

picture *picture::transformed(const transform& t)
{
  picture *pic = new picture;

  list<drawElement*>::iterator p;
  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    pic->append((*p)->transformed(t));
  }

  return pic;
}


} // namespace camp

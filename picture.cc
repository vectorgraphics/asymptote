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

    double offset=-1.5*72;
    double hoffset=offset+printerOffset.getx();
    double voffset=offset+printerOffset.gety();
    
    dcmd << "dvips -E -O " << hoffset << "bp," << voffset << "bp";
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
			  const string& outputformat, bool wait)
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
  const string AltViewers[]={PSViewer,"ggv","ghostview","gsview"};
  static size_t nViewers=sizeof(AltViewers)/sizeof(string);
  static size_t iviewer=0;
  if(view && !deconstruct) {
    if(epsformat || pdfformat) {
      if (first || !interact::virtualEOF) {
	first=false;
	status=-1;
	while(status == -1 && iviewer < nViewers) {
	  ostringstream cmd;
	  cmd << AltViewers[iviewer];
	  if(AltViewers[iviewer] == "gv") cmd << " -nowatch";
	  cmd << " " << outname;
	  status=System(cmd,false,wait,&pid,iviewer+1 == nViewers);
	  if(status == -1) ++iviewer;
	}
	if(status) return false;
	// Tell gv it should reread the file.
      } else if(AltViewers[iviewer] == "gv") kill(pid,SIGHUP);
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
  string outputformat=(format == "" ? outformat : format);
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
  // Avoid negative bounding box coordinates
    bboxshift=pair(-b.left,-b.bottom);
    bpos.shift(bboxshift);
  }
  
  if(bpos.right <= bpos.left && bpos.top <= bpos.bottom) { // null picture
    unlink(outname.c_str());
    return true;
  }
  
  static const double pdfoffset=2.0;
  if(!labels && outputformat == "pdf") {
    bpos.right += pdfoffset;
    bpos.top += pdfoffset;
  }
  
  string texname=auxname(prefix,"tex");
  texfile *tex=NULL;
  bool status = true;
  
  if(labels) {
    tex=new texfile(texname,b);
    list<drawElement*>::iterator p;
    for (p = nodes.begin(); p != nodes.end(); ++p) {
      assert(*p);
      if (!(*p)->setup(tex))
	status = false;
    }
  
    tex->prologue();
  }
  
  list<drawElement*>::iterator layerp=nodes.begin();
  list<drawElement*>::iterator p=layerp;
  unsigned int layer=0;
  list<string> psnameStack;
  
  while(p != nodes.end()) {
    ostringstream buf;
    buf << prefix << layer;
    string psname=labels ? auxname(buf.str(),"ps") : epsname;
    psnameStack.push_back(psname);
    psfile out(psname,bpos,bboxshift);
    out.prologue();
  
    if(labels) tex->beginlayer(psname);
  
    for(; p != nodes.end(); ++p) {
      assert(*p);
      if(labels && (*p)->islayer()) break;
      if(!(*p)->draw(&out))
	status = false;
    }
    out.epilogue();
  
    if(status && labels) {
      for (p=layerp; p != nodes.end(); ++p) {
	if((*p)->islayer()) {
	  tex->endlayer();
	  layerp=++p;
	  layer++;
	  break;
	}
	assert(*p);
	if (!(*p)->write(tex))
	  status = false;
      }
    }    
  }
  
  if(status) {
    if(labels) {
      tex->epilogue();
      status=texprocess(texname,epsname,prefix);
      if(!keep) {
	list<string>::iterator p;
	for(p=psnameStack.begin(); p != psnameStack.end(); ++p)
	  unlink(p->c_str());
      }
    }
    status=postprocess(epsname,outname,outputformat,wait);
  }
  
  if(labels) delete tex;
  
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

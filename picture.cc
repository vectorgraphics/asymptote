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
			 const string& prefix, const bbox& bpos) 
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
    
    // Magic dvips offsets:
    double hoffset=-128.5;
    double voffset=-126;
    
    if(pdfformat || bottomOrigin)
      voffset += max(11.0*72.0-ceil(bpos.top-bpos.bottom),0.0);
    if(!pdfformat) {
      hoffset += postscriptOffset.getx();
      voffset -= postscriptOffset.gety();
    }

    string psname=auxname(prefix,"ps");
    ostringstream dcmd;
    dcmd << "dvips -O " << hoffset << "bp," << voffset << "bp";
    if(verbose <= 2) dcmd << " -q";
    dcmd << " -o " << psname << " " << dviname;
    status=System(dcmd);
    
    ifstream fin(psname.c_str());
    ofstream fout(outname.c_str());
    string s;
    while(getline(fin,s)) {
      if(s.find("%%BoundingBox:") == 0) 
	BoundingBox(fout,bpos);
      else fout << s << endl;
    }
    
    if(!keep) { // Delete temporary files.
      unlink("texput.log");
      unlink(texname.c_str());
      unlink(dviname.c_str());
      unlink(psname.c_str());
      unlink(auxname(prefix,"aux").c_str());
      unlink(auxname(prefix,"log").c_str());
    }
  }
  if(status) return false;
  return true;
}

bool picture::postprocess(const string& epsname, const string& outname,
			  const string& outputformat, bool wait,
			  const bbox& bpos)
{
  int status=0;
  ostringstream cmd;
  
  if(!epsformat) {
    if(pdfformat) cmd << "gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop"
		      << " -dDEVICEWIDTHPOINTS=" 
		      << ceil(bpos.right-bpos.left+1.0)
		      << " -dDEVICEHEIGHTPOINTS=" 
		      << ceil(bpos.top-bpos.bottom+1.0)
		      << " -sOutputFile=" << outname << " " << epsname;
    else {
      double res=deconstruct*72.0;
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
  if(view && !deconstruct) {
    if(epsformat || pdfformat) {
      static int pid;
      static bool first=true;
      static const string PSViewers[]={PSViewer,"ggv","ghostview","gsview"};
      static const string PDFViewers[]={PDFViewer,"acroread","xpdf"};
      static const size_t nPSViewers=sizeof(PSViewers)/sizeof(string);
      static const size_t nPDFViewers=sizeof(PDFViewers)/sizeof(string);
      const string *Viewers=pdfformat ? PDFViewers : PSViewers;
      const size_t nViewers=pdfformat ? nPDFViewers : nPSViewers;
      size_t iViewer=0;
      if (first || !interact::virtualEOF) {
	first=false;
	status=-1;
	while(status == -1 && iViewer < nViewers) {
	  ostringstream cmd;
	  cmd << Viewers[iViewer];
	  if(Viewers[iViewer] == "gv") cmd << " -nowatch";
	  cmd << " " << outname;
	  status=System(cmd,false,wait,&pid,iViewer+1 == nViewers);
	  if(status == -1) ++iViewer;
	}
	if(status) return false;
	// Tell gv it should reread the file.
      } else if(Viewers[iViewer] == "gv") kill(pid,SIGHUP);
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
  string outputformat=format == "" ? outformat : format;
  epsformat=outputformat == "" || outputformat == "eps";
  pdfformat=outputformat == "pdf";
  tgifformat=outputformat == "tgif";
  
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
      
  string outname=tgifformat ? "."+buildname(prefix,"gif") :
    buildname(prefix,outputformat);
  string epsname=epsformat ? outname : auxname(prefix,"eps");
  
  bounds();
  
  bbox bcopy=b;
  
  static double fuzz=1.0;
  bcopy.right += fuzz;
  bcopy.left -= fuzz;
  bcopy.top += fuzz;
  bcopy.bottom -= fuzz;
  
  bbox bpos=bcopy;
  
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
    bboxshift=pair(-bpos.left,-bpos.bottom);
    if(!pdfformat) {
      bboxshift += postscriptOffset;
      if(!bottomOrigin)
	bboxshift += pair(0.0,max(11.0*72.0-ceil(bpos.top-bpos.bottom),0.0));
    }
    bpos.shift(bboxshift);
  }
  
  if(bpos.right <= bpos.left && bpos.top <= bpos.bottom) { // null picture
    unlink(outname.c_str());
    return true;
  }
  
  string texname=auxname(prefix,"tex");
  texfile *tex=NULL;
  bool status = true;
  
  if(labels) {
    tex=new texfile(texname,bcopy);
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
    buf << prefix << "_" << layer;
    string psname=labels ? buildname(buf.str(),"ps") : epsname;
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
      status=texprocess(texname,epsname,prefix,bpos);
      if(!keep) {
	list<string>::iterator p;
	for(p=psnameStack.begin(); p != psnameStack.end(); ++p)
	  unlink(p->c_str());
      }
    }
    status=postprocess(epsname,outname,outputformat,wait,bpos);
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

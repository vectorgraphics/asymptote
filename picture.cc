/*****
 * picture.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a picture as a list of drawElements and handles its output to 
 * PostScript. 
 *****/

#include <csignal>

#include "errormsg.h"
#include "picture.h"
#include "util.h"
#include "settings.h"
#include "interact.h"

using std::ifstream;
using std::ofstream;

using namespace settings;

namespace camp {

const char *texpathmessage() {
  ostringstream buf;
  buf << "the directory containing your " << getSetting<string>("tex")
      << " engine (" << texengine() << ")";
  return strcpy(new char[buf.str().size()+1],buf.str().c_str());
}
  
texstream tex; // Bi-directional pipe to latex (to find label bbox)

void texstream::pipeclose() {
  iopipestream::pipeclose();
  if(!getSetting<bool>("keep")) {
    unlink("texput.log");
    unlink("texput.out");
    unlink("texput.aux");
    if(settings::pdf(texengine()))
      unlink("texput.pdf");
  }
}

picture::~picture()
{
}

bool picture::epsformat,picture::pdfformat,picture::xasyformat, picture::pdf;
bool picture::Labels;
double picture::paperWidth,picture::paperHeight;
  
void picture::enclose(drawElement *begin, drawElement *end)
{
  assert(begin);
  assert(end);
  nodes.push_front(begin);
  lastnumber=0;
  for(nodelist::iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if((*p)->islayer()) {
      nodes.insert(p,end);
      ++p;
     while(p != nodes.end() && (*p)->islayer()) ++p;
     if(p == nodes.end()) return;
     nodes.insert(p,begin);
    }
  }
  nodes.push_back(end);
}

// Insert at beginning of picture.
void picture::prepend(drawElement *p)
{
  assert(p);
  nodes.push_front(p);
  lastnumber=0;
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
  copy(pic.nodes.begin(), pic.nodes.end(), back_inserter(nodes));
}

// Insert picture pic at beginning of picture.
void picture::prepend(picture &pic)
{
  if (&pic == this) return;
  
  copy(pic.nodes.begin(), pic.nodes.end(), inserter(nodes, nodes.begin()));
  lastnumber=0;
}

bool picture::havelabels()
{
  size_t n=nodes.size();
  if(n > lastnumber && !labels && getSetting<string>("tex") != "none") {
    // Check to see if there are any labels yet
    nodelist::iterator p=nodes.begin();
    for(size_t i=0; i < lastnumber; ++i) ++p;
    for(; p != nodes.end(); ++p) {
      assert(*p);
      if((*p)->islabel())
        labels=true;
    }
  }
  return labels;
}

bbox picture::bounds()
{
  size_t n=nodes.size();
  if(n == lastnumber) return b;
  
  if(lastnumber == 0) { // Maybe these should be put into a structure.
    b=bbox();
    labelbounds.clear();
    bboxstack.clear();
  }
  
  if(havelabels()) texinit();
  
  nodelist::iterator p=nodes.begin();
  for(size_t i=0; i < lastnumber; ++i) ++p;
  for(; p != nodes.end(); ++p) {
    assert(*p);
    (*p)->bounds(b,tex,labelbounds,bboxstack);
  }

  lastnumber=n;
  return b;
}

void picture::texinit()
{
  drawElement::lastpen=pen(initialpen);
  // Output any new texpreamble commands
  if(tex.isopen()) {
    if(TeXpipepreamble.empty()) return;
    texpreamble(tex,TeXpipepreamble);
    TeXpipepreamble.clear();
    return;
  }
  
  ostringstream cmd;
  cmd << "'" << texprogram() << "'" << " \\scrollmode";
  tex.open(cmd.str().c_str(),"texpath",texpathmessage());
  tex.wait("\n*");
  tex << "\n";
  texdocumentclass(tex,true);
  
  texdefines(tex,TeXpreamble,true);
  TeXpipepreamble.clear();
}
  
bool picture::texprocess(const string& texname, const string& outname,
			 const string& prefix, const pair& bboxshift) 
{
  int status=0;
  ifstream outfile;
  
  outfile.open(texname.c_str());
  if(outfile) {
    outfile.close();
    ostringstream cmd;
    cmd << "'" << texprogram() << "'"
	<< " \\nonstopmode\\input '" << texname << "'";
    bool quiet=verbose <= 1;
    status=System(cmd,quiet ? 1 : 0,"texpath",texpathmessage());
    if(!status && getSetting<bool>("twice"))
      status=System(cmd,quiet ? 1 : 0,"texpath",texpathmessage());
    if(status) {
      if(quiet) System(cmd,0);
      return false;
    }
    
    if(!pdf) {
      string dviname=auxname(prefix,"dvi");
      string psname=auxname(prefix,"ps");
    
      double height=b.top-b.bottom+1.0;
    
      // Magic dvips offsets:
      double hoffset=-128.4;
      double vertical=height;
      if(!latex(getSetting<string>("tex"))) vertical += 2.0;
      double voffset=(vertical < 13.0) ? -137.8+vertical : -124.8;

      hoffset += b.left+bboxshift.getx();
      voffset += paperHeight-height-b.bottom-bboxshift.gety();
    
      ostringstream dcmd;
      dcmd << "'" << getSetting<string>("dvips") << "' "
	   << getSetting<string>("dvipsOptions") << " -R "
	   << " -O " << hoffset << "bp," << voffset << "bp"
	   << " -T " << paperWidth << "bp," << paperHeight << "bp";
      if(verbose <= 1) dcmd << " -q";
      dcmd << " -o '" << psname << "' '" << dviname << "'";
      status=System(dcmd,0,true,"dvips");
    
      ifstream fin(psname.c_str());
      psfile fout(outname,false);
    
      string s;
      bool first=true;
      transform t=shift(bboxshift);
      if(T) t=t*(*T);
      bool shift=(t != identity());
      string beginspecial="TeXDict begin @defspecial";
      string endspecial="@fedspecial end";
      while(getline(fin,s)) {
	if(s.find("%%DocumentPaperSizes:") == 0) continue;
	if(s.find("%!PS-Adobe-") == 0) {
	  fout.verbatimline("%!PS-Adobe-3.0 EPSF-3.0");
	} else if(first && s.find("%%BoundingBox:") == 0) {
	  bbox box=b;
	  box.shift(bboxshift);
	  if(verbose > 2) BoundingBox(cout,box);
	  fout.BoundingBox(box);
	  first=false;
	} else if(shift && s.find(beginspecial) == 0) {
	  fout.verbatimline(s);
	  fout.gsave();
	  fout.concat(t);
	} else if(shift && s.find(endspecial) == 0) {
	  fout.grestore();
	  fout.verbatimline(s);
	} else
	  fout.verbatimline(s);
      }
      if(!getSetting<bool>("keep")) { // Delete temporary files.
	unlink(dviname.c_str());
	unlink(psname.c_str());
      }
    }
      
    if(!getSetting<bool>("keep")) { // Delete temporary files.
      unlink(texname.c_str());
      if(!getSetting<bool>("keepaux")) unlink(auxname(prefix,"aux").c_str());
      unlink(auxname(prefix,"log").c_str());
      unlink(auxname(prefix,"out").c_str());
    }
  }
  if(status) return false;
  return true;
}

int picture::epstopdf(const string& epsname, const string& pdfname)
{
  ostringstream cmd;
  
  cmd << "'" << getSetting<string>("gs")
      << "' -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop"
      << " -dCompatibilityLevel=1.4";
  if(!getSetting<bool>("autorotate"))
    cmd << " -dAutoRotatePages=/None";
  cmd << " -g" << max(ceil(paperWidth),1.0) << "x" << max(ceil(paperHeight),1.0)
      << " -dDEVICEWIDTHPOINTS=" << max(b.right-b.left,3.0)
      << " -dDEVICEHEIGHTPOINTS=" << max(b.top-b.bottom,3.0)
      << " -sOutputFile='" << pdfname << "' '" << epsname << "'";
  return System(cmd,0,true,"gs","Ghostscript");
}
  
std::map<CONST string,int> pids;

bool picture::postprocess(const string& prename, const string& outname,
			  const string& outputformat, bool wait, bool view)
{
  int status=0;
  
  if((pdf && Labels) || !epsformat) {
    if(pdfformat) {
      if(pdf && Labels) status=rename(prename.c_str(),outname.c_str());
      else status=epstopdf(prename,outname);
    } else {
      ostringstream cmd;
      double expand=2.0;
      double res=(xasyformat ? getSetting<double>("deconstruct") : expand)*
	72.0;
      cmd << "'" << getSetting<string>("convert") 
	  << "' -density " << res << "x" << res;
      if(!xasyformat) cmd << " +antialias -geometry " << 100.0/expand << "%x";
      cmd << " '" << (pdf ? "pdf:" : "eps:") << prename << "'";
      if(xasyformat) cmd << " -transparent white gif";
      else cmd << " " << outputformat;
      cmd << ":'" << outname << "'";
      status=System(cmd,0,true,"convert");
    }
    if(!getSetting<bool>("keep")) unlink(prename.c_str());
  }
  if(status != 0) return false;
  
  if(verbose > (xasyformat ? 1 : 0)) 
    cout << "Wrote " << outname << endl;
  if(settings::view() && view) {
    if(epsformat || pdfformat) {
      // Check to see if there is an existing viewer for this outname.
      std::map<CONST string,int>::iterator p=pids.find(outname);
      bool running=(p != pids.end());
      string Viewer=pdfformat ? getSetting<string>("pdfviewer") :
	getSetting<string>("psviewer");
      int pid;
      if(running) {
	pid=p->second;
	if(interact::interactive && pid)
	  running=(waitpid(pid, &status, WNOHANG) != pid);
      }
	
      if(!running) {
	ostringstream cmd;
	cmd << "'" << Viewer << "'";
	cmd << " '" << outname << "'";
	status=System(cmd,0,wait,
		      pdfformat ? "pdfviewer" : "psviewer",
		      pdfformat ? "your PDF viewer" : "your PostScript viewer",
		      &pid);
	pids[outname]=pid;
	if(status != 0) return false;
      } else if(Viewer == "gv") kill(pid,SIGHUP); // Tell gv to reread file.
    } else {
      ostringstream cmd;
      cmd << "'" << getSetting<string>("display") << "' '"
	  << outname << "'";
      string application="your "+outputformat+" viewer";
      status=System(cmd,0,wait,"display",application.c_str());
      if(status != 0) return false;
    }
  }
  
  return true;
}

bool picture::shipout(picture *preamble, const string& Prefix,
		      const string& format, bool wait, bool view, bool Delete)
{
  bounds();
  
  bool TeXmode=getSetting<bool>("inlinetex") && 
    getSetting<string>("tex") != "none";
  Labels=labels || TeXmode;
  
  pdf=settings::pdf(getSetting<string>("tex"));
  
  bool standardout=Prefix == "-";
  string prefix=standardout ? "out" : Prefix;
  string preformat=nativeformat();
  string outputformat=format.empty() ? defaultformat() : format;
  epsformat=outputformat == "eps";
  pdfformat=outputformat == "pdf";
  xasyformat=outputformat == "<xasy>";
  string outname=xasyformat ? "."+buildname(prefix,"gif") :
    (standardout ? "-" : buildname(prefix,outputformat,"",!global()));
  string epsname=epsformat ? (standardout ? "" : outname) :
    auxname(prefix,"eps");
  string prename=((epsformat && !pdf) || !Labels) ? epsname : 
    auxname(prefix,preformat);
  double deconstruct=getSetting<double>("deconstruct");
  
  static ofstream bboxout;
  
  if(b.empty && !Labels) { // Output a null file
    bbox b;
    b.left=b.bottom=0;
    b.right=b.top=1;
    psfile out(epsname,false);
    out.prologue(b);
    out.epilogue();
    out.close();
    if(deconstruct && !xasyformat) {
      if(bboxout) bboxout.close();
      ShipoutNumber++;
      return true;
    }
    return postprocess(epsname,outname,outputformat,wait,view);
  }
  
  if(deconstruct) {
    bool signal=getSetting<bool>("signal");
    if(!bboxout.is_open()) {
      bboxout.open(("."+buildname(prefix,"box")).c_str());	
      bboxout << (xasyformat ? deconstruct : 0) << newl;
    }
    if(xasyformat) {
      bbox bscaled=b;
      bscaled *= deconstruct;
      bboxout << bscaled << endl;
      if(signal) bboxout.close();
      if(Delete) {
	unlink(outname.c_str());
	return false;
      }
    } else {
      if(bboxout) bboxout.close();
      if(settings::view() && view) {
	ostringstream cmd;
	string Python=getSetting<string>("python");
	if(Python != "") cmd << "'" << Python << "' ";
	cmd << "'" << getSetting<string>("xasy") << "' " 
	    << buildname(prefix) << " " << ShipoutNumber << " "
	    << buildname(settings::outname());
	int status=System(cmd,0,true,Python != "" ? "python" : "xasy");
	if(status != 0) return false;
      }
      ShipoutNumber++;
      return true;
    }
  }

      
  SetPageDimensions();
  
  paperWidth=getSetting<double>("paperwidth");
  paperHeight=getSetting<double>("paperheight");
  int origin=getSetting<int>("align");
    
  pair bboxshift=(origin == ZERO && !pdfformat) ?
    pair(0.0,0.0) : pair(-b.left,-b.bottom);
  if(!pdfformat) {
    bboxshift += getSetting<pair>("offset");
    if(origin != ZERO && origin != BOTTOM) {
      double yexcess=max(paperHeight-(b.top-b.bottom+1.0),0.0);
      if(origin == TOP) bboxshift += pair(0.0,yexcess);
      else {
	double xexcess=max(paperWidth-(b.right-b.left+1.0),0.0);
	bboxshift += pair(0.5*xexcess,0.5*yexcess);
      }
    }
  }
  
  bool status = true;
  
  string texname;
  texfile *tex=NULL;
  
  if(Labels) {
    spaceToUnderscore(prefix);
    texname=auxname(prefix,"tex");
    tex=new texfile(texname,b);
    tex->prologue();
  }
  
  nodelist::iterator layerp=nodes.begin();
  nodelist::iterator p=layerp;
  unsigned int layer=0;
  mem::list<string> psnameStack;
  
  bbox bshift=b;
  
  while(p != nodes.end()) {
    string psname,pdfname;
    if(Labels) {
      ostringstream buf;
      buf << prefix << "_" << layer;
      psname=buildname(buf.str(),"eps");
      if(pdf) pdfname=buildname(buf.str(),"pdf");
    } else {
      psname=epsname;
      bshift.shift(bboxshift);
    }
    psnameStack.push_back(psname);
    if(pdf) psnameStack.push_back(pdfname);
    psfile out(psname,pdfformat);
    out.prologue(bshift);
  
    if(Labels) tex->beginlayer(pdf ? pdfname : psname);
    else {
      out.gsave();
      out.translate(bboxshift);
    }
  
    if(preamble) {
      // Postscript preamble.
      nodelist Nodes=preamble->nodes;
      nodelist::iterator P=Nodes.begin();
      if(P != Nodes.end()) {
	out.resetpen();
	for(; P != Nodes.end(); ++P) {
	  assert(*P);
	  if(!(*P)->draw(&out))
	    status = false;
	}
      }
    }
    out.resetpen();
    
    for(; p != nodes.end(); ++p) {
      assert(*p);
      if(Labels && (*p)->islayer()) break;
      if(!(*p)->draw(&out))
	status = false;
    }
    if(!Labels) out.grestore();
    
    out.epilogue();
    out.close();
    
    if(Labels) {
      tex->resetpen();
      if(status) {
	if(pdf && !b.empty)
	  status=(epstopdf(psname,pdfname) == 0);
	
	if(status) {
	  for (p=layerp; p != nodes.end(); ++p) {
	    if((*p)->islayer()) {
	      tex->endlayer();
	      layerp=++p;
	      layer++;
	      break;
	    }
	    assert(*p);
	    if(!(*p)->write(tex))
	      status = false;
	  }
	}
      }    
    }
  }
  
  if(status) {
    if(TeXmode) {
      if(Labels && verbose > 0) cout << "Wrote " << texname << endl;
      delete tex;
    } else {
      if(Labels) {
	tex->epilogue();
	status=texprocess(texname,prename,prefix,bboxshift);
	delete tex;
	if(!getSetting<bool>("keep")) {
	  for(mem::list<string>::iterator p=psnameStack.begin();
	      p != psnameStack.end(); ++p)
	    unlink(p->c_str());
	}
      }
      if(status)
	status=postprocess(prename,outname,outputformat,wait,view);
    }
  }
  
  if(!status) reportError("shipout failed");
    
  return status;
}

picture *picture::transformed(const transform& t)
{
  picture *pic = new picture;

  nodelist::iterator p;
  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    pic->append((*p)->transformed(t));
  }
  pic->T=new transform(T ? t*(*T) : t);

  return pic;
}


} // namespace camp

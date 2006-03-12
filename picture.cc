
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

double pageWidth, pageHeight;

namespace camp {

string texready=string("(Please type a command or say `\\end')\n*");
texstream tex; // Bi-directional pipe to latex (to find label bbox)

void texstream::pipeclose() {
  iopipestream::pipeclose();
  if (!getSetting<bool>("keep")) {
    unlink("texput.log");
    unlink("texput.aux");
  }
}

picture::~picture()
{
}

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
  if(n > lastnumber && !labels && getSetting<bool>("tex")) {
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
  if(TeXinitialized) {
    if(TeXpipepreamble.empty()) return;
    texpreamble(tex,TeXpipepreamble);
    TeXpipepreamble.clear();
    return;
  }
  
  tex.open(getSetting<mem::string>("latex").c_str(),"latex","latex");
  texdocumentclass(tex,true);
  
  texdefines(tex,TeXpipepreamble,true);
  TeXpipepreamble.clear();

  tex << "\n";
  tex.wait(texready.c_str(),"! ");
  TeXinitialized=true;
}
  
bool picture::texprocess(const string& texname, const string& outname,
			 const string& prefix, const bbox& bpos) 
{
  int status=0;
  ifstream outfile;
  
  outfile.open(texname.c_str());
  if(outfile) {
    outfile.close();
    ostringstream cmd;
    cmd << "'" << getSetting<mem::string>("latex") << "'"
	<< " \\scrollmode\\input " << texname;
    bool quiet=verbose <= 1;
    status=System(cmd,quiet,"latex");
    if(status) {
      if(quiet) status=System(cmd,false,"latex");
      return false;
    }
    
    string dviname=auxname(prefix,"dvi");
    string psname=auxname(prefix,"ps");
    
    double height=bpos.top-bpos.bottom;
    double width=bpos.right-bpos.left;
    
    // Magic dvips offsets:
    double hoffset=-128.0;
    double voffset=(height < 11.0) ? -137.0+height : -126.0;
    
    int origin=getSetting<int>("align");

    if(origin != ZERO) {
      if(pdfformat || origin == BOTTOM) {
	voffset += max(pageHeight-(height+1.0),0.0);
      } else if(origin == CENTER) {
	hoffset += 0.5*max(pageWidth-(width+1.0),0.0);
	voffset += 0.5*max(pageHeight-(height+1.0),0.0);
      }
    }
    
    if(!pdfformat) {
      hoffset += getSetting<pair>("offset").getx();
      voffset -= getSetting<pair>("offset").gety();
    }

    ostringstream dcmd;
    dcmd << "'" << getSetting<mem::string>("dvips") << "' -R -t " 
	 << getSetting<mem::string>("papertype") 
	 << "size -O " << hoffset << "bp," << voffset << "bp";
    if(verbose <= 1) dcmd << " -q";
    dcmd << " -o " << psname << " " << dviname;
    status=System(dcmd,false,true,"dvips");
    
    bbox bcopy=bpos;
    double hfuzz=0.1;
    double vfuzz=0.2;
    if(origin == CENTER || origin == TOP) {
      hfuzz *= 2.0; vfuzz *= 2.0;
    }
    
    bcopy.left -= hfuzz;
    bcopy.right += hfuzz;
    
    bcopy.bottom -= vfuzz;
    bcopy.top += vfuzz;
    
    ifstream fin(psname.c_str());
    ofstream *Fout=NULL;
    ostream *fout=(outname == "") ? &cout :
      Fout=new ofstream(outname.c_str());
    
    string s;
    bool first=true;
    while(getline(fin,s)) {
      if(s.find("%%DocumentPaperSizes:") == 0) continue;
      if(first && s.find("%%BoundingBox:") == 0) {
	if(verbose > 2) BoundingBox(cout,bpos);
	BoundingBox(*fout,bcopy);
	first=false;
      } else *fout << s << endl;
    }
    if(Fout) {
      if(!Fout->good()) {
	ostringstream msg;
	msg << "Cannot write to " << outname.c_str();
	reportError(msg);
      }
      delete Fout;
    }
    
    if(!getSetting<bool>("keep")) { // Delete temporary files.
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
			  const string& outputformat, bool wait, bool quiet,
			  const bbox& bpos)
{
  int status=0;
  ostringstream cmd;
  
  if(!epsformat) {
    if(pdfformat) {
      cmd << "'" << getSetting<mem::string>("gs")
	  << "' -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop"
	  << " -dAutoRotatePages=/None "
	  << " -dDEVICEWIDTHPOINTS=" 
	  << bpos.right-bpos.left+1.0
	  << " -dDEVICEHEIGHTPOINTS=" 
	  << bpos.top-bpos.bottom+1.0
	  << " -sOutputFile=" << outname << " " << epsname;
      status=System(cmd,false,true,"gs","Ghostscript");
    } else {
      double expand=2.0;
      double res=(tgifformat ? getSetting<double>("deconstruct") : expand)*
	72.0;
      cmd << "'" << getSetting<mem::string>("convert") 
	  << "' -density " << res << "x" << res;
      if(!tgifformat) cmd << " +antialias -geometry " << 100.0/expand << "%x";
      cmd << " eps:" << epsname;
      if(tgifformat) cmd << " -transparent white gif";
      else cmd << " " << outputformat;
      cmd << ":" << outname;
      status=System(cmd,false,true,"convert");
    }
    if(!getSetting<bool>("keep")) unlink(epsname.c_str());
  }
  if(status != 0) return false;
  
  if(verbose > (tgifformat ? 1 : 0)) 
    cout << "Wrote " << outname << endl;
  if(view() && !quiet) {
    if(epsformat || pdfformat) {
      static int pid=0;
      static string lastoutname;
      string Viewer=pdfformat ? getSetting<mem::string>("pdfviewer") :
	getSetting<mem::string>("psviewer");
      bool restart=false;
      if(interact::interactive && pid)
	restart=(waitpid(pid, &status, WNOHANG) == pid);

      if (!interact::virtualEOF || outname != lastoutname || restart) {
	if(!wait) lastoutname=outname;
	ostringstream cmd;
	cmd << "'" << Viewer << "'";
	if(Viewer == "gv" && interact::interactive)
	  cmd << " -nowatch";
	cmd << " " << outname;
	status=System(cmd,false,wait,
		      pdfformat ? "pdfviewer" : "psviewer",
		      pdfformat ? "your PDF viewer" : "your PostScript viewer",
		      &pid);
	if(status != 0) return false;
      } else if(Viewer == "gv") kill(pid,SIGHUP); // Tell gv to reread file.
    } else {
      ostringstream cmd;
      cmd << "'" << getSetting<mem::string>("display") << "' " << outname;
      string application="your "+outputformat+" viewer";
      status=System(cmd,false,wait,"display",application.c_str());
      if(status != 0) return false;
    }
  }
  
  return true;
}

bool picture::shipout(picture *preamble, const string& Prefix,
		      const string& format, bool wait, bool quiet, bool Delete)
{
  bool standardout=Prefix == "-";
  string prefix=standardout ? "out" : Prefix;
  checkFormatString(format);
  string outputformat=format.empty() ? 
    (string)getSetting<mem::string>("outformat") : format;
  epsformat=outputformat.empty() || outputformat == "eps";
  pdfformat=outputformat == "pdf";
  tgifformat=outputformat == "tgif";
  string outname=tgifformat ? "."+buildname(prefix,"gif") :
    (standardout ? "-" : buildname(prefix,outputformat,"",false));
  string epsname=epsformat ? (standardout ? "" : outname) :
    auxname(prefix,"eps");
  double deconstruct=getSetting<double>("deconstruct");
  
  bounds();
  
  static ofstream bboxout;
  
  if(null()) { // Output a null file
    bbox b;
    b.left=b.bottom=0;
    b.right=b.top=1;
    psfile out(epsname,b,0);
    out.prologue();
    out.epilogue();
    if(deconstruct && !tgifformat) {
      if(bboxout) bboxout.close();
      ShipoutNumber++;
      return true;
    }
    return postprocess(epsname,outname,outputformat,wait,quiet,b);
  }
  
  if(deconstruct && !tgifformat) {
    if(bboxout) bboxout.close();
    if(view()) {
      ostringstream cmd;
      string Python=getSetting<mem::string>("python");
      if(Python != "") cmd << "'" << Python << "' ";
      cmd << "'" << getSetting<mem::string>("xasy") << "' " 
	  << buildname(prefix) << " " << ShipoutNumber << " "
	  << buildname(getSetting<mem::string>("outname"));
      int status=System(cmd,false,true,Python != "" ? "python" : "xasy");
      if(status != 0) return false;
    }
    ShipoutNumber++;
    return true;
  }
      
  bbox bpos=b;
  
  bool TeXmode=getSetting<bool>("inlinetex") && getSetting<bool>("tex");
  bool Labels=labels || TeXmode;
  
  if(deconstruct) {
      if(!bboxout.is_open()) {
	bboxout.open(("."+buildname(prefix,"box")).c_str());	
	bboxout << deconstruct << endl;
      }
      bbox bscaled=b;
      bscaled *= deconstruct;
      bboxout << bscaled << endl;
      if(Delete) {
	unlink(outname.c_str());
	return false;
      }
  }
  
  GetPageDimensions(pageWidth,pageHeight);
  
  // Avoid negative bounding box coordinates
  int origin=getSetting<int>("align");
  bboxshift=origin == ZERO ? 0.0 : pair(-bpos.left,-bpos.bottom);
  if(!pdfformat) {
    bboxshift += getSetting<pair>("offset");
    if(!(origin == BOTTOM || origin == ZERO)) {
      double yexcess=max(pageHeight-(bpos.top-bpos.bottom),0.0);
      if(origin == TOP) bboxshift += pair(0.5,yexcess-0.5);
      else {
	double xexcess=max(pageWidth-(bpos.right-bpos.left),0.0);
	bboxshift += 0.5*pair(xexcess,yexcess);
      }
    }
  }
  bpos.shift(bboxshift);
  
  bool status = true;
  
  string texname=auxname(prefix,"tex");
  texfile *tex=NULL;
  
  if(Labels) {
    tex=new texfile(texname,b);
    tex->prologue();
  }
  
  nodelist::iterator layerp=nodes.begin();
  nodelist::iterator p=layerp;
  unsigned int layer=0;
  std::list<string> psnameStack;
  
  while(p != nodes.end()) {
    ostringstream buf;
    buf << prefix << "_" << layer;
    string psname=Labels ? buildname(buf.str(),"eps") : epsname;
    psnameStack.push_back(psname);
    psfile out(psname,bpos,bboxshift);
    out.prologue();
  
    if(Labels) tex->beginlayer(psname);
  
    if(preamble) {
      // Postscript preamble.
      nodelist Nodes=preamble->nodes;
      nodelist::iterator P=Nodes.begin();
      if(P != Nodes.end()) {
	out.resetpen();
	for(; P != Nodes.end(); ++P) {
	  assert(*P);
	  out.raw(true);
	  if(!(*P)->draw(&out))
	    status = false;
	  out.raw(false);
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
    out.epilogue();
  
    if(status && Labels) {
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
  
  if(status) {
    if(TeXmode) {
      if(verbose > 0) cout << "Wrote " << texname << endl;
    } else {
      if(labels) {
	tex->epilogue();
	status=texprocess(texname,epsname,prefix,bpos);
	if(!getSetting<bool>("keep"))
	  for(std::list<string>::iterator p=psnameStack.begin();
	      p != psnameStack.end(); ++p)
	    unlink(p->c_str());
      }
      if(status)
	status=postprocess(epsname,outname,outputformat,wait,quiet,bpos);
    }
  }
  
  if(!status) reportError("shipout failed");
    
  delete tex;
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

  return pic;
}


} // namespace camp

/*****
 * picture.cc
 * Andy Hammerlindl 2002/06/06
 *
 * Stores a picture as a list of drawElements and handles its output to 
 * PostScript. 
 *****/

#include "errormsg.h"
#include "picture.h"
#include "util.h"
#include "settings.h"
#include "interact.h"
#include "drawverbatim.h"

using std::ifstream;
using std::ofstream;

using namespace settings;
using namespace gl;

texstream::~texstream() {
  string name=stripFile(outname())+"texput.";
  unlink((name+"aux").c_str());
  unlink((name+"log").c_str());
  unlink((name+"out").c_str());
  if(settings::pdf(getSetting<string>("tex")))
    unlink((name+"pdf").c_str());
}

namespace camp {

const char *texpathmessage() {
  ostringstream buf;
  buf << "the directory containing your " << getSetting<string>("tex")
      << " engine (" << texcommand() << ")";
  return Strdup(buf.str());
}
  
picture::~picture()
{
}

bool picture::epsformat,picture::pdfformat,picture::xobject, picture::pdf;
bool picture::Labels;
double picture::paperWidth,picture::paperHeight;
  
void picture::enclose(drawElement *begin, drawElement *end)
{
  assert(begin);
  assert(end);
  nodes.push_front(begin);
  lastnumber=0;
  lastnumber3=0;
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
  lastnumber3=0;
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
  lastnumber3=0;
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
      if((*p)->islabel()) {
        labels=true;
	break;
      }
    }
  }
  return labels;
}

bool picture::have3D()
{
  for(nodelist::iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if((*p)->is3D())
      return true;
  }
  return false;
}

bbox picture::bounds()
{
  size_t n=nodes.size();
  if(n == lastnumber) return b_cached;
  
  if(lastnumber == 0) { // Maybe these should be put into a structure.
    b_cached=bbox();
    labelbounds.clear();
    bboxstack.clear();
  }
  
  if(havelabels()) texinit();
  
  nodelist::iterator p=nodes.begin();
  for(size_t i=0; i < lastnumber; ++i) ++p;
  for(; p != nodes.end(); ++p) {
    assert(*p);
    (*p)->bounds(b_cached,processData().tex,labelbounds,bboxstack);
    
     // Optimization for interpreters with fixed stack limits.
    if((*p)->endclip()) {
      nodelist::iterator q=p;
      if(q != nodes.begin()) {
	--q;
	assert(*q);
	if((*q)->endclip())
	  (*q)->save(false);
      }
    }
  }

  lastnumber=n;
  return b_cached;
}

bbox3 picture::bounds3()
{
  size_t n=nodes.size();
  if(n == lastnumber3) return b3;
  
  if(lastnumber3 == 0)
    b3=bbox3();
  
  nodelist::iterator p=nodes.begin();
  for(size_t i=0; i < lastnumber3; ++i) ++p;
  for(; p != nodes.end(); ++p) {
    assert(*p);
    (*p)->bounds(b3);
  }

  lastnumber3=n;
  return b3;
}
  
pair picture::bounds(double (*m)(double, double),
		     double (*x)(const triple&, double*),
		     double (*y)(const triple&, double*),
		     double *t)
{
  bool first=true;
  pair b;
  for(nodelist::const_iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->bounds(b,m,x,y,t,first);
  }
  return b;
}
  
void picture::texinit()
{
  drawElement::lastpen=pen(initialpen);
  processDataStruct &pd=processData();
  // Output any new texpreamble commands
  if(pd.tex.isopen()) {
    if(pd.TeXpipepreamble.empty()) return;
    texpreamble(pd.tex,pd.TeXpipepreamble,false);
    pd.TeXpipepreamble.clear();
    return;
  }
  
  string name=stripFile(outname())+"texput.aux";
  const char *cname=name.c_str();
  ofstream writeable(cname);
  if(!writeable)
    reportError("Cannot write to "+name);
  else
    writeable.close();
  unlink(cname);
  
  ostringstream cmd;
  cmd << texprogram() << " \\scrollmode";
  pd.tex.open(cmd.str().c_str(),"texpath",texpathmessage());
  pd.tex.wait("\n*");
  pd.tex << "\n";
  texdocumentclass(pd.tex,true);
  
  texdefines(pd.tex,pd.TeXpreamble,true);
  pd.TeXpipepreamble.clear();
}
  
bool picture::texprocess(const string& texname, const string& outname,
			 const string& prefix, const pair& bboxshift) 
{
  int status=0;
  ifstream outfile;
  
  outfile.open(texname.c_str());
  if(outfile) {
    outfile.close();
    string aux=auxname(prefix,"aux");
    unlink(aux.c_str());
    string program=texprogram();
    ostringstream cmd;
    cmd << program << " \\nonstopmode\\input '" << texname << "'";
    bool quiet=verbose <= 1;
    status=System(cmd,quiet ? 1 : 0,"texpath",texpathmessage());
    if(!status && getSetting<bool>("twice"))
      status=System(cmd,quiet ? 1 : 0,"texpath",texpathmessage());
    if(status) {
      if(quiet) {
	ostringstream cmd;
	cmd << program << " \\scrollmode\\input '" << texname << "'";
	System(cmd,0);
      }
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
      
      string dvipsrc=getSetting<string>("dir");
      if(dvipsrc.empty()) dvipsrc=systemDir;
      dvipsrc += dirsep+"nopapersize.ps";
      setenv("DVIPSRC",dvipsrc.c_str(),1);
      dcmd << "'" << getSetting<string>("dvips") << "' -R -Pdownload35 -D600"
	   << " -O" << hoffset << "bp," << voffset << "bp"
	   << " -T" << paperWidth << "bp," << paperHeight << "bp "
	   << getSetting<string>("dvipsOptions") << " -tnopapersize";
      if(verbose <= 1) dcmd << " -q";
      dcmd << " -o '" << psname << "' '" << dviname << "'";
      status=System(dcmd,0,true,"dvips");
      if(status != 0) return false;
    
      ifstream fin(psname.c_str());
      psfile fout(outname,false);
    
      string s;
      bool first=true;
      transform t=shift(bboxshift)*T;
      bool shift=!t.isIdentity();
      string beginspecial="TeXDict begin @defspecial";
      string endspecial="@fedspecial end";
      while(getline(fin,s)) {
	if(s.find("%%DocumentPaperSizes:") == 0) continue;
	if(s.find("%!PS-Adobe-") == 0) {
	  fout.header();
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
      if(!getSetting<bool>("keepaux"))
	unlink(aux.c_str());
      unlink(auxname(prefix,"log").c_str());
      unlink(auxname(prefix,"out").c_str());
    }
    if(status == 0) return true;
  }
  return false;
}

int picture::epstopdf(const string& epsname, const string& pdfname)
{
  ostringstream cmd;
  
  cmd << "'" << getSetting<string>("gs")
      << "' -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dEPSCrop"
      << " -dSubsetFonts=true -dEmbedAllFonts=true -dMaxSubsetPct=100"
      << " -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.4";
  if(safe)
    cmd << " -dSAFER";
  if(!getSetting<bool>("autorotate"))
    cmd << " -dAutoRotatePages=/None";
  cmd << " -g" << max(ceil(paperWidth),1.0) << "x" << max(ceil(paperHeight),1.0)
      << " -dDEVICEWIDTHPOINTS=" << max(b.right-b.left,3.0)
      << " -dDEVICEHEIGHTPOINTS=" << max(b.top-b.bottom,3.0)
      << " " << getSetting<string>("gsOptions")
      << " -sOutputFile='" << pdfname << "' '" << epsname << "'";

  string dir=stripFile(pdfname);
  char *oldPath=NULL;
  if(!dir.empty()) {
    oldPath=getPath();
    setPath(dir.c_str());
  }
  int status=System(cmd,0,true,"gs","Ghostscript");
  if(oldPath != NULL)
    setPath(oldPath);
  return status;
}
  
bool picture::reloadPDF(const string& Viewer, const string& outname) const 
{
  static bool needReload=true;
  static bool haveReload=false;
  
  // Send javascript code to redraw picture.
  picture f;
  string name=getPath()+string("/")+outname;
  f.append(new drawVerbatim(TeX,"\\ \\pdfannot width 0pt height 0pt { /AA << /PO << /S /JavaScript /JS (try{reload('"+
			    name+"');} catch(e) {} closeDoc(this);) >> >> }"));
  string reloadprefix="reload";
  if(needReload) {
    needReload=false;
    string texengine=getSetting<string>("tex");
    Setting("tex")=string("pdflatex");
    haveReload=f.shipout(NULL,reloadprefix,"pdf",0.0,false,false);
    Setting("tex")=texengine;
  }
  if(haveReload) {
    ostringstream cmd;
    cmd << "'" << Viewer << "' ";
    string pdfreloadOptions=getSetting<string>("pdfreloadOptions");
    if(!pdfreloadOptions.empty())
      cmd << pdfreloadOptions << " ";
    cmd << "'" << reloadprefix << ".pdf'";
    System(cmd,0,false);
  }
  return true;
}		
  
  
bool picture::postprocess(const string& prename, const string& outname,
			  const string& outputformat, double magnification,
			  bool wait, bool view)
{
  static mem::map<CONST string,int> pids;
  int status=0;
  
  if((pdf && Labels) || !epsformat) {
    if(pdfformat) {
      if(pdf && Labels) status=rename(prename.c_str(),outname.c_str());
      else status=epstopdf(prename,outname);
    } else {
      ostringstream cmd;
      double render=fabs(getSetting<double>("render"));
      if(render == 0) render=1.0;
      double expand=getSetting<Int>("antialias");
      if(expand < 2.0) expand=1.0;
      double res=expand*render*72.0;
      cmd << "'" << getSetting<string>("convert") 
	  << "' -alpha Off -density " << res << "x" << res;
      if(expand == 1.0)
	cmd << " +antialias";
      cmd << " -geometry " << 100.0/expand << "%x"
	  << " " << getSetting<string>("convertOptions")
	  << " '" << nativeformat()+":" << prename << "'"
          << " '" << outputformat << ":" << outname << "'";
      status=System(cmd,0,true,"convert");
    }
    if(!getSetting<bool>("keep")) unlink(prename.c_str());
  }
  if(status != 0) return false;
  
  if(verbose > 0)
    cout << "Wrote " << outname << endl;
  bool View=settings::view() && view;
  if(View) {
    if(epsformat || pdfformat) {
      // Check to see if there is an existing viewer for this outname.
      mem::map<CONST string,int>::iterator p=pids.find(outname);
      bool running=(p != pids.end());
      string Viewer=pdfformat ? getSetting<string>("pdfviewer") :
	getSetting<string>("psviewer");
      int pid;
      if(running) {
	pid=p->second;
	if(pid)
	  running=(waitpid(pid, &status, WNOHANG) != pid);
      }
	
      bool pdfreload=pdfformat && getSetting<bool>("pdfreload");
      if(running) {
	// Tell gv/acroread to reread file.	  
	if(Viewer == "gv") kill(pid,SIGHUP);
	else if(pdfreload) reloadPDF(Viewer,outname);
      } else {
	ostringstream cmd;
	cmd << "'" << Viewer << "' ";
	string viewerOptions=getSetting<string>(pdfformat ? 
						"pdfviewerOptions" : 
						"psviewerOptions");
	if(!viewerOptions.empty())
	  cmd << viewerOptions << " ";
	cmd << "'" << outname << "'";
	status=System(cmd,0,wait,
		      pdfformat ? "pdfviewer" : "psviewer",
		      pdfformat ? "your PDF viewer" : "your PostScript viewer",
		      &pid);
	if(status != 0) return false;
	
	if(!wait) pids[outname]=pid;

	if(pdfreload) {
	  // Work around race conditions in acroread initialization script
	  usleep(getSetting<Int>("pdfreloaddelay"));
	  // Only reload if pdf viewer process is already running.
	  if(waitpid(pid, &status, WNOHANG) == pid)
	    reloadPDF(Viewer,outname);
	}
      }
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

string Outname(const string& prefix, const string& outputformat,
	       bool standardout)
{
  return standardout ? "-" : buildname(prefix,outputformat,"");
}

bool picture::shipout(picture *preamble, const string& Prefix,
		      const string& format, double magnification,
		      bool wait, bool view)
{
  b=bounds();
  
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
  xobject=magnification > 0;
  string outname=Outname(prefix,outputformat,standardout);
  string epsname=epsformat ? (standardout ? "" : outname) :
    auxname(prefix,"eps");
  string prename=((epsformat && !pdf) || !Labels) ? epsname : 
    auxname(prefix,preformat);
  
  if((b.empty && !Labels)) { // Output a null file
    bbox b;
    b.left=b.bottom=0;
    b.right=b.top=xobject ? 18 : 1;
    psfile out(epsname,false);
    out.prologue(b);
    out.epilogue();
    out.close();
    return postprocess(epsname,outname,outputformat,1.0,wait,view);
  }
  
  bool pngxformat=xobject && getSetting<string>("xformat") == "png";
  if(pngxformat) {
    // Work around half-pixel bounding box bug in Ghostscript pngalpha driver
    double fuzz=0.5/magnification;
    b.top += fuzz;
    b.right += fuzz;
    b.bottom -= fuzz;
  }
    
  SetPageDimensions();
  
  paperWidth=getSetting<double>("paperwidth");
  paperHeight=getSetting<double>("paperheight");
  Int origin=getSetting<Int>("align");
    
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
  unsigned layer=0;
  mem::list<string> psnameStack;
  
  bbox bshift=b;
  
  transparency=false;
  
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
    
    if(out.Transparency())
      transparency=true;
    
    if(Labels) {
      tex->resetpen();
      if(status) {
	if(pdf && !b.empty) {
	  status=(epstopdf(psname,pdfname) == 0);
	  if(!getSetting<bool>("keep")) unlink(psname.c_str());
	}
	
	if(status) {
	  for (p=layerp; p != nodes.end(); ++p) {
	    assert(*p);
	    if(!(*p)->write(tex,b))
	      status = false;
	    if((*p)->islayer()) {
	      tex->endlayer();
	      layerp=++p;
	      layer++;
	      break;
	    }
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
      if(status) {
	if(xobject) {
	  if(transparency && pngxformat)
	    status=(epstopdf(prename,Outname(prefix,"pdf",standardout)) == 0);
	} else
	  status=postprocess(prename,outname,outputformat,magnification,wait,
			     view);
      }
    }
  }
  
  if(!status) reportError("shipout failed");
    
  return true;
}

// render viewport with width x height pixels.
void picture::render(GLUnurbs *nurb, double size2,
		     const triple& Min, const triple& Max,
		     double perspective, bool transparent) const
{
  for(nodelist::const_iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->render(nurb,size2,Min,Max,perspective,transparent);
  }
}
  
struct Communicate : public gc {
  string prefix;
  picture* pic;
  string format;
  double width;
  double height;
  double angle;
  triple m;
  triple M;
  size_t nlights;
  triple *lights;
  double *diffuse;
  double *ambient;
  double *specular;
  bool viewportlighting;
  bool view;
};

Communicate com;

void glrenderWrapper()
{
#ifdef HAVE_LIBGLUT  
  glrender(com.prefix,com.pic,com.format,com.width,com.height,com.angle,
	   com.m,com.M,com.nlights,com.lights,com.diffuse,com.ambient,
	   com.specular,com.viewportlighting,com.view);
#endif  
}

void hold(bool View) 
{
#ifdef HAVE_LIBGLUT  
#ifdef HAVE_LIBPTHREAD
  if(glthread) {
    if(!View)
      wait(readySignal,readyLock);
  
    if(!interact::interactive)
      wait(quitSignal,quitLock);
  }
#endif  
#endif  
}

extern bool glinitialize;

bool picture::shipout3(const string& prefix, const string& format,
		       double width, double height,
		       double angle, const triple& m, const triple& M,
		       size_t nlights, triple *lights, double *diffuse,
		       double *ambient, double *specular, bool viewportlighting,
		       bool view)
{
#ifdef HAVE_LIBGLUT
  bounds3();
  
  for(nodelist::const_iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->displacement();
  }

  const string outputformat=format.empty() ? 
    getSetting<string>("outformat") : format;
  bool View=settings::view() && view;
  static int oldpid=0;
  
  if(glthread) {
#ifdef HAVE_LIBPTHREAD
    if(gl::initialize) {
      gl::initialize=false;
      com.prefix=prefix;
      com.pic=this;
      com.format=outputformat;
      com.width=width;
      com.height=height;
      com.angle=angle;
      com.m=m;
      com.M=M;
      com.nlights=nlights;
      com.lights=lights;
      com.diffuse=diffuse;
      com.ambient=ambient;
      com.specular=specular;
      com.viewportlighting=viewportlighting;
      com.view=View;
      wait(initSignal,initLock);
      hold(View);
      return true;
    }
#endif
  } else {
    int pid=fork();
    if(pid == -1)
      camp::reportError("Cannot fork process");
    if(pid != 0)  {
      oldpid=pid;
      waitpid(pid,NULL,interact::interactive ? WNOHANG : 0);
      return true;
    }
  }
  
  glrender(prefix,this,outputformat,width,height,angle,m,M,
	   nlights,lights,diffuse,ambient,specular,viewportlighting,View,
	   oldpid);
  hold(View);
#else
  reportError("Cannot render image; please install glut, run ./configure, and recompile"); 
#endif
  return false;
}

bool picture::shipout3(const string& prefix)
{
  bounds3();
  bool status = true;
  
  string prcname=buildname(prefix,"prc");
  prcfile prc(prcname);
  for(nodelist::iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    if(!(*p)->write(&prc))
      status = false;
  }
  if(status)
    status=prc.finish();
    
  if(!status) reportError("shipout3 failed");
    
  if(verbose > 0) cout << "Wrote " << prcname << endl;
  
  return true;
}

picture *picture::transformed(const transform& t)
{
  picture *pic = new picture;

  nodelist::iterator p;
  for (p = nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    pic->append((*p)->transformed(t));
  }
  pic->T=transform(t*T);

  return pic;
}

picture *picture::transformed(const vm::array& t)
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

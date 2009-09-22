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
  string texengine=getSetting<string>("tex");
  bool context=settings::context(texengine);
  string name;
  if(!context) 
    name=stripFile(outname());
  name += "texput.";
  unlink((name+"aux").c_str());
  unlink((name+"log").c_str());
  unlink((name+"out").c_str());
  if(settings::pdf(texengine))
    unlink((name+"pdf").c_str());
  if(context) {
    unlink((name+"tex").c_str());
    unlink((name+"top").c_str());
    unlink((name+"tua").c_str());
    unlink((name+"tui").c_str());
  }
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

bool picture::epsformat,picture::pdfformat, picture::svgformat;
bool picture::xobject, picture::pdf;
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
  
pair picture::ratio(double (*m)(double, double))
{
  bool first=true;
  pair b;
  for(nodelist::const_iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->ratio(b,m,first);
  }
  return b;
}
  
void texinit()
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
  
  bool context=settings::context(getSetting<string>("tex"));
  string dir=stripFile(outname());
  string logname;
  if(!context) logname=dir;
  logname += "texput.log";
  const char *cname=logname.c_str();
  ofstream writeable(cname);
  if(!writeable)
    reportError("Cannot write to "+logname);
  else
    writeable.close();
  unlink(cname);
  
  mem::vector<string> cmd;
  cmd.push_back(texprogram());
  if(context) {
    // Create a null texput.tex file as a portable way of tricking ConTeXt
    // into entering interactive mode (pending the implementation of --pipe).
    string texput="texput.tex";
    ofstream(texput.c_str());
    cmd.push_back("--scrollmode");
    cmd.push_back(texput);
  } else {
    if(!dir.empty()) 
      cmd.push_back("-output-directory="+dir);
    cmd.push_back("\\scrollmode");
  }
  
  pd.tex.open(cmd,"texpath",texpathmessage());
  pd.tex.wait("\n*");
  pd.tex << "\n";
  texdocumentclass(pd.tex,true);
  
  texdefines(pd.tex,pd.TeXpreamble,true);
  pd.TeXpipepreamble.clear();
}
  
int opentex(const string& texname, const string& prefix) 
{
  string aux=auxname(prefix,"aux");
  unlink(aux.c_str());
  bool context=settings::context(getSetting<string>("tex"));
  mem::vector<string> cmd;
  cmd.push_back(texprogram());
  if(context) {
    cmd.push_back("--nonstopmode");
    cmd.push_back(texname);
  } else {
    string dir=stripFile(texname);
    if(!dir.empty()) 
      cmd.push_back("-output-directory="+dir);
    cmd.push_back("\\nonstopmode\\input");
    cmd.push_back(stripDir(texname));
  }
    
  bool quiet=verbose <= 1;
  int status=System(cmd,quiet ? 1 : 0,true,"texpath",texpathmessage());
  if(!status && getSetting<bool>("twice"))
    status=System(cmd,quiet ? 1 : 0,true,"texpath",texpathmessage());
  if(status) {
    if(quiet) {
      cmd[1]=context ? "--scrollmode" : "\\scrollmode\\input";
      System(cmd,0);
    }
  }
  return status;
}


bool picture::texprocess(const string& texname, const string& outname,
                         const string& prefix, const pair& bboxshift) 
{
  int status=0;
  ifstream outfile;
  
  outfile.open(texname.c_str());
  if(outfile) {
    outfile.close();
    
    if(opentex(texname,prefix)) return false;
    
    string texengine=getSetting<string>("tex");
    string dviname=auxname(prefix,"dvi");
    
    if(svgformat) {
      mem::vector<string> cmd;
      cmd.push_back(getSetting<string>("dvisvgm"));
      cmd.push_back("-n");
      cmd.push_back("--verbosity=3");
      push_split(cmd,getSetting<string>("dvisvgmOptions"));
      cmd.push_back("-o"+outname);
      cmd.push_back(dviname);
      status=System(cmd,0,true,"dvisvgm");
      if(status != 0) return false;
    } else {
      if(!pdf) {
        string psname=auxname(prefix,"ps");
        double height=b.top-b.bottom+1.0;
    
        // Magic dvips offsets:
        double hoffset=-128.4;
        double vertical=height;
        if(!latex(texengine)) vertical += 2.0;
        double voffset=(vertical < 13.0) ? -137.8+vertical : -124.8;

        hoffset += b.left+bboxshift.getx();
        voffset += paperHeight-height-b.bottom-bboxshift.gety();
    
        string dvipsrc=getSetting<string>("dir");
        if(dvipsrc.empty()) dvipsrc=systemDir;
        dvipsrc += dirsep+"nopapersize.ps";
        setenv("DVIPSRC",dvipsrc.c_str(),1);
        string papertype=getSetting<string>("papertype") == "letter" ?
          "letterSize" : "a4size";
        mem::vector<string> dcmd;
        dcmd.push_back(getSetting<string>("dvips"));
        dcmd.push_back("-R");
        dcmd.push_back("-Pdownload35");
        dcmd.push_back("-D600");
        dcmd.push_back("-O"+String(hoffset)+"bp,"+String(voffset)+"bp");
        dcmd.push_back("-T"+String(paperWidth)+"bp,"+String(paperHeight)+"bp");
        push_split(dcmd,getSetting<string>("dvipsOptions"));
        dcmd.push_back("-t"+papertype);
        if(verbose <= 1) dcmd.push_back("-q");
        dcmd.push_back("-o"+psname);
        dcmd.push_back(dviname);
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
    }
      
    if(!getSetting<bool>("keep")) { // Delete temporary files.
      unlink(texname.c_str());
      if(!getSetting<bool>("keepaux"))
        unlink(auxname(prefix,"aux").c_str());
      unlink(auxname(prefix,"log").c_str());
      unlink(auxname(prefix,"out").c_str());
      if(settings::context(texengine)) {
        unlink(auxname(prefix,"top").c_str());
        unlink(auxname(prefix,"tua").c_str());
        unlink(auxname(prefix,"tuc").c_str());
        unlink(auxname(prefix,"tui").c_str());
        unlink(auxname(prefix,"tuo").c_str());
      }
    }
    if(status == 0) return true;
  }
  return false;
}

int picture::epstopdf(const string& epsname, const string& pdfname)
{
  mem::vector<string> cmd;
  cmd.push_back(getSetting<string>("gs"));
  cmd.push_back("-q");
  cmd.push_back("-dNOPAUSE");
  cmd.push_back("-dBATCH");
  cmd.push_back("-sDEVICE=pdfwrite");
  cmd.push_back("-dEPSCrop");
  cmd.push_back("-dSubsetFonts=true");
  cmd.push_back("-dEmbedAllFonts=true");
  cmd.push_back("-dMaxSubsetPct=100");
  cmd.push_back("-dPDFSETTINGS=/prepress");
  cmd.push_back("-dCompatibilityLevel=1.4");
  if(safe)
    cmd.push_back("-dSAFER");
  if(!getSetting<bool>("autorotate"))
    cmd.push_back("-dAutoRotatePages=/None");
  cmd.push_back("-g"+String(max(ceil(paperWidth),1.0))+"x"+
                String(max(ceil(paperHeight),1.0)));
  cmd.push_back("-dDEVICEWIDTHPOINTS="+String(max(b.right-b.left,3.0)));
  cmd.push_back("-dDEVICEHEIGHTPOINTS="+String(max(b.top-b.bottom,3.0)));
  push_split(cmd,getSetting<string>("gsOptions"));
  cmd.push_back("-sOutputFile="+stripDir(pdfname));
  cmd.push_back(stripDir(epsname));

  char *oldPath=NULL;
  string dir=stripFile(pdfname);
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
    mem::vector<string> cmd;
    push_command(cmd,Viewer);
    string pdfreloadOptions=getSetting<string>("pdfreloadOptions");
    if(!pdfreloadOptions.empty())
      cmd.push_back(pdfreloadOptions);
    cmd.push_back(reloadprefix+".pdf");
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
      if(pdf && Labels) {
        status=rename(prename.c_str(),outname.c_str());
        if(status != 0)
          reportError("Cannot rename "+prename+" to "+outname);
      } else status=epstopdf(prename,outname);
    } else if(!svgformat) {
      mem::vector<string> cmd;
      double render=fabs(getSetting<double>("render"));
      if(render == 0) render=1.0;
      double expand=getSetting<Int>("antialias");
      if(expand < 2.0) expand=1.0;
      double res=expand*render*72.0;
      cmd.push_back(getSetting<string>("convert")); 
      cmd.push_back("-alpha");
      cmd.push_back("Off");
      cmd.push_back("-density");
      cmd.push_back(String(res)+"x"+String(res));
      if(expand == 1.0)
        cmd.push_back("+antialias");
      cmd.push_back("-geometry");
      cmd.push_back(String(100.0/expand)+"%x");
      push_split(cmd,getSetting<string>("convertOptions"));
      cmd.push_back(nativeformat()+":"+prename);
      cmd.push_back(outputformat+":"+outname);
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
        mem::vector<string> cmd;
        push_command(cmd,Viewer);
        string viewerOptions=getSetting<string>(pdfformat ? 
                                                "pdfviewerOptions" : 
                                                "psviewerOptions");
        if(!viewerOptions.empty())
          push_split(cmd,viewerOptions);
        cmd.push_back(outname);
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
      mem::vector<string> cmd;
      push_command(cmd,getSetting<string>("display"));
      cmd.push_back(outname);
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
  
  string texengine=getSetting<string>("tex");
  pdf=settings::pdf(texengine);
  
  bool standardout=Prefix == "-";
  string prefix=standardout ? "out" : Prefix;
  string preformat=nativeformat();
  string outputformat=format.empty() ? defaultformat() : format;
  epsformat=outputformat == "eps";
  pdfformat=outputformat == "pdf";
  svgformat=outputformat == "svg" && !pdf &&
    (!have3D() || getSetting<double>("render") == 0.0);
  
  xobject=magnification > 0;
  string outname=Outname(prefix,outputformat,standardout);
  string epsname=epsformat ? (standardout ? "" : outname) :
    auxname(prefix,"eps");
  
  Labels=labels || TeXmode || svgformat;
  
  if(Labels)
    spaceToUnderscore(prefix);
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
  
  if(xobject) {
    // Work around half-pixel bounding box bug in Ghostscript pngalpha driver
    double fuzz=0.5/magnification;
    b.top += fuzz;
    b.right += fuzz;
    b.bottom -= fuzz;
  }
    
  SetPageDimensions();
  
  paperWidth=getSetting<double>("paperwidth");
  paperHeight=getSetting<double>("paperheight");
  string origin=getSetting<string>("align");
    
  pair bboxshift=(origin == "Z" && !pdfformat) ?
    pair(0.0,0.0) : pair(-b.left,-b.bottom);
  if(!pdfformat) {
    bboxshift += getSetting<pair>("offset");
    if(origin != "Z" && origin != "B") {
      double yexcess=max(paperHeight-(b.top-b.bottom+1.0),0.0);
      if(origin == "T") bboxshift += pair(0.0,yexcess);
      else {
        double xexcess=max(paperWidth-(b.right-b.left+1.0),0.0);
        bboxshift += pair(0.5*xexcess,0.5*yexcess);
      }
    }
  }
  
  bool status=true;
  
  string texname;
  texfile *tex=NULL;
  
  if(Labels) {
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
  
    if(!Labels) {
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
          (*P)->draw(&out);
        }
      }
    }
    out.resetpen();
    
    bool postscript=false;
    for(; p != nodes.end(); ++p) {
      assert(*p);
      if(Labels && (*p)->islayer()) break;
      postscript |= (*p)->draw(&out);
    }
    
    if(Labels) {
      tex->beginlayer(pdf ? pdfname : psname,postscript);
    } else out.grestore();
    
    out.epilogue();
    out.close();
    
    if(out.Transparency())
      transparency=true;
    
    if(Labels) {
      tex->resetpen();
      if(pdf && !b.empty) {
        status=(epstopdf(psname,pdfname) == 0);
        if(!getSetting<bool>("keep")) unlink(psname.c_str());
      }
        
      if(status) {
        for (p=layerp; p != nodes.end(); ++p) {
          assert(*p);
          (*p)->write(tex,b);
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
  
  bool context=settings::context(texengine);
  if(status) {
    if(TeXmode) {
      if(Labels && verbose > 0) cout << "Wrote " << texname << endl;
      delete tex;
    } else {
      if(Labels) {
        tex->epilogue();
        if(context) prefix=stripDir(prefix);
        status=texprocess(texname,svgformat ? outname : prename,prefix,
                          bboxshift);
        delete tex;
        if(!getSetting<bool>("keep")) {
          for(mem::list<string>::iterator p=psnameStack.begin();
              p != psnameStack.end(); ++p)
            unlink(p->c_str());
        }
      }
      if(status) {
        if(xobject) {
          if(transparency)
            status=(epstopdf(prename,Outname(prefix,"pdf",standardout)) == 0);
        } else {
          if(context) prename=stripDir(prename);
          status=postprocess(prename,outname,outputformat,magnification,wait,
                             view);
        }
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
  double zoom;
  triple m;
  triple M;
  pair shift;
  double *t;
  double *background;
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
#ifdef HAVE_LIBGL  
#ifdef HAVE_LIBPTHREAD
  wait(initSignal,initLock);
  endwait(initSignal,initLock);
#endif  
  glrender(com.prefix,com.pic,com.format,com.width,com.height,com.angle,
           com.zoom,com.m,com.M,com.shift,com.t,com.background,com.nlights,
           com.lights,com.diffuse,com.ambient,com.specular,com.viewportlighting,
           com.view);
#endif  
}

bool picture::shipout3(const string& prefix, const string& format,
                       double width, double height, double angle, double zoom,
                       const triple& m, const triple& M, const pair& shift,
                       double *t, double *background, size_t nlights,
                       triple *lights, double *diffuse, double *ambient,
                       double *specular, bool viewportlighting, bool view)
{
#ifdef HAVE_LIBGL
  bounds3();
  
  for(nodelist::const_iterator p=nodes.begin(); p != nodes.end(); ++p) {
    assert(*p);
    (*p)->displacement();
  }

  const string outputformat=format.empty() ? 
    getSetting<string>("outformat") : format;
  bool View=settings::view() && view;
  static int oldpid=0;
  bool Wait=!interact::interactive || !View;
  
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
      com.zoom=zoom;
      com.m=m;
      com.M=M;
      com.shift=shift;
      com.t=t;
      com.background=background;
      com.nlights=nlights;
      com.lights=lights;
      com.diffuse=diffuse;
      com.ambient=ambient;
      com.specular=specular;
      com.viewportlighting=viewportlighting;
      com.view=View;
#ifdef HAVE_LIBPTHREAD
      if(Wait)
        pthread_mutex_lock(&readyLock);
      wait(initSignal,initLock);
      endwait(initSignal,initLock);
      static bool initialize=true;
      if(initialize) {
        wait(initSignal,initLock);
        endwait(initSignal,initLock);
        initialize=false;
      }
      if(Wait) {
        pthread_cond_wait(&readySignal,&readyLock);
        pthread_mutex_unlock(&readyLock);
      }
#endif  
      return true;
    }
#ifdef HAVE_LIBPTHREAD
    if(Wait)
      pthread_mutex_lock(&readyLock);
#endif  
#endif
  } else {
    int pid=fork();
    if(pid == -1)
      camp::reportError("Cannot fork process");
    if(pid != 0)  {
      oldpid=pid;
      waitpid(pid,NULL,interact::interactive && View ? WNOHANG : 0);
      return true;
    }
  }
  
  glrender(prefix,this,outputformat,width,height,angle,zoom,m,M,shift,t,
           background,nlights,lights,diffuse,ambient,specular,viewportlighting,
           View,oldpid);
#ifdef HAVE_LIBPTHREAD
  if(glthread && Wait) {
    pthread_cond_wait(&readySignal,&readyLock);
    pthread_mutex_unlock(&readyLock);
  }
  return true;
#endif  
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
    (*p)->write(&prc);
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

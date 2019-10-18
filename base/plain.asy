/*****
 * plain.asy
 * Andy Hammerlindl and John Bowman 2004/08/19
 *
 * A package for general purpose drawing, with automatic sizing of pictures.
 *
 *****/

access settings;

if(settings.command != "") {
  string s=settings.command;
  settings.command="";
  settings.multipleView=settings.batchView=settings.interactiveView;
  _eval(s+";",false,true);
  exit();
}

include plain_constants;

access version;             
if(version.VERSION != VERSION) {
  warning("version","using possibly incompatible version "+
          version.VERSION+" of plain.asy"+'\n');
  nowarn("version");
}
   
include plain_strings;
include plain_pens;
include plain_paths;
include plain_filldraw;
include plain_margins;
include plain_picture;
include plain_Label;
include plain_shipout;
include plain_arcs;
include plain_boxes;
include plain_markers;
include plain_arrows;
include plain_debugger;

typedef void exitfcn();

void updatefunction()
{
  implicitshipout=true;
  if(!currentpicture.uptodate) shipout();
  implicitshipout=false;
}

void exitfunction()
{
  implicitshipout=true;
  if(!currentpicture.empty())
    shipout();
  implicitshipout=false;
}

atupdate(updatefunction);
atexit(exitfunction);

// A restore thunk is a function, that when called, restores the graphics state
// to what it was when the restore thunk was created.
typedef void restoreThunk();
typedef restoreThunk saveFunction();
saveFunction[] saveFunctions={};

// When save is called, this will be redefined to do the corresponding restore.
void restore()
{
  warning("nomatchingsave","restore called with no matching save");
}

void addSaveFunction(saveFunction s)
{
  saveFunctions.push(s);
}

restoreThunk buildRestoreThunk()
{
  // Call the save functions in reverse order, storing their restore thunks.
  restoreThunk[] thunks={};
  for (int i=saveFunctions.length-1; i >= 0; --i)
    thunks.push(saveFunctions[i]());

  return new void() {
    // Call the restore thunks in an order matching the saves.
    for (int i=thunks.length-1; i >= 0; --i)
      thunks[i]();
  };
}

// Add the default save function.
addSaveFunction(new restoreThunk () {
    pen defaultpen=defaultpen();
    pen p=currentpen;
    picture pic=currentpicture.copy();
    restoreThunk r=restore;
    return new void() {
      defaultpen(defaultpen);
      currentpen=p;
      currentpicture=pic;
      currentpicture.uptodate=false;
      restore=r;
    };
  });

// Save the current state, so that restore will put things back in that state.
restoreThunk save() 
{
  return restore=buildRestoreThunk();
}

void restoredefaults()
{
  warning("nomatchingsavedefaults",
          "restoredefaults called with no matching savedefaults");
}

restoreThunk buildRestoreDefaults()
{
  pen defaultpen=defaultpen();
  exitfcn atupdate=atupdate();
  exitfcn atexit=atexit();
  restoreThunk r=restoredefaults;
  return new void() {
    defaultpen(defaultpen);
    atupdate(atupdate);
    atexit(atexit);
    restoredefaults=r;
  };
}

// Save the current state, so that restore will put things back in that state.
restoreThunk savedefaults() 
{
  return restoredefaults=buildRestoreDefaults();
}

void initdefaults()
{
  savedefaults();
  resetdefaultpen();
  atupdate(null);
  atexit(null);
}

// Return the sequence n,...m
int[] sequence(int n, int m)
{
  return sequence(new int(int x){return x;},m-n+1)+n;
}

int[] reverse(int n) {return sequence(new int(int x){return n-1-x;},n);}
bool[] reverse(bool[] a) {return a[reverse(a.length)];}
int[] reverse(int[] a) {return a[reverse(a.length)];}
real[] reverse(real[] a) {return a[reverse(a.length)];}
pair[] reverse(pair[] a) {return a[reverse(a.length)];}
triple[] reverse(triple[] a) {return a[reverse(a.length)];}
string[] reverse(string[] a) {return a[reverse(a.length)];}

// Return a uniform partition dividing [a,b] into n subintervals.
real[] uniform(real a, real b, int n)
{
  if(n <= 0) return new real[];
  return a+(b-a)/n*sequence(n+1);
}

void eval(string s, bool embedded=false)
{
  if(!embedded) initdefaults();
  _eval(s+";",embedded);
  if(!embedded) restoredefaults();
}

void eval(code s, bool embedded=false)
{
  if(!embedded) initdefaults();
  _eval(s,embedded);
  if(!embedded) restoredefaults();
}

// Evaluate user command line option.
void usersetting()
{
  eval(settings.user,true);
}

string stripsuffix(string f, string suffix=".asy") 
{
  int n=rfind(f,suffix);
  if(n != -1) f=erase(f,n,-1);
  return f;
}

string outdirectory()
{
  return stripfile(outprefix());
}

// Conditionally process each file name in array s in a new environment.
void asy(string format, bool overwrite=false ... string[] s)
{
  for(string f : s) {
    f=stripsuffix(f);
    string suffix="."+format;
    string fsuffix=stripdirectory(f+suffix);
    if(overwrite || error(input(outdirectory()+fsuffix,check=false))) {
      string outformat=settings.outformat;
      bool interactiveView=settings.interactiveView;
      bool batchView=settings.batchView;
      settings.outformat=format;
      settings.interactiveView=false;
      settings.batchView=false;
      string outname=outname();
      delete(outname+"_"+".aux");
      eval("import \""+f+"\" as dummy");
      rename(stripsuffix(outname)+suffix,fsuffix);
      settings.outformat=outformat;
      settings.interactiveView=interactiveView;
      settings.batchView=batchView;
    }
  }
}

void beep()
{
  write('\7',flush);
}

struct processtime {
  real user;
  real system;
}

struct cputime {
  processtime parent;
  processtime child;
  processtime change;
}

cputime cputime() 
{
  static processtime last;
  real [] a=_cputime();
  cputime cputime;
  cputime.parent.user=a[0];
  cputime.parent.system=a[1];
  cputime.child.user=a[2];
  cputime.child.system=a[3];
  real user=a[0]+a[2];
  real system=a[1]+a[3];
  cputime.change.user=user-last.user;
  cputime.change.system=system-last.system;
  last.user=user;
  last.system=system;
  return cputime;
}

string cputimeformat="%#.2f";

void write(file file, string s="", cputime c, string format=cputimeformat,
           suffix suffix=none)
{
  write(file,s,
        format(format,c.change.user)+"u "+
        format(format,c.change.system)+"s "+
        format(format,c.parent.user+c.child.user)+"U "+
        format(format,c.parent.system+c.child.system)+"S ",suffix);
}

void write(string s="", cputime c, string format=cputimeformat,
           suffix suffix=endl)
{
  write(stdout,s,c,format,suffix);
}

if(settings.autoimport != "") {
  string s=settings.autoimport;
  settings.autoimport="";
  eval("import \""+s+"\" as dummy",true);
  atupdate(updatefunction);
  atexit(exitfunction);
  settings.autoimport=s;
}

cputime();

void nosetpagesize()
{
  static bool initialized=false;
  if(!initialized && latex()) {
    // Portably pass nosetpagesize option to graphicx package.
    texpreamble("\usepackage{ifluatex}\ifluatex
\ifx\pdfpagewidth\undefined\let\pdfpagewidth\paperwidth\fi
\ifx\pdfpageheight\undefined\let\pdfpageheight\paperheight\fi\else
\let\paperwidthsave\paperwidth\let\paperwidth\undefined
\usepackage{graphicx}
\let\paperwidth\paperwidthsave\fi");
    initialized=true;
  }
}

nosetpagesize();

if(settings.tex == "luatex")
  texpreamble("\input luatex85.sty");

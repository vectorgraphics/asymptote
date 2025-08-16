/*****
 *plain.asy
 *Andy Hammerlindl and John Bowman 2004/08/19
 *
 *A package for general purpose drawing, with automatic sizing of pictures.
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
include plain_arcs;
include plain_boxes;
include plain_shipout;
include plain_markers;
include plain_arrows;
include plain_debugger;

real RELEASE=(real) split(VERSION,"-")[0];

using exitfcn=void();

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
using restoreThunk=void();
using saveFunction=restoreThunk();
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
  for(int i=saveFunctions.length-1; i >= 0; --i)
    thunks.push(saveFunctions[i]());

  return new void() {
    // Call the restore thunks in an order matching the saves.
    for(int i=thunks.length-1; i >= 0; --i)
      thunks[i]();
  };
}

// Add the default save function.
addSaveFunction(new restoreThunk() {
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

// Return the sequence n,...,m
int[] sequence(int n, int m)
{
  return n+sequence(m-n+1);
}

// Return the sequence n,...,m skipping by skip
int[] sequence(int n, int m, int skip)
{
  return n+skip*sequence((m-n)#skip+1);
}

from collections.iter(T=int) access
    Iter_T as Iter_int,
    Iterable_T as Iterable_int,
    Iterable;

Iterable_int range(int n, int m, int skip=1) {
  Iterable_int result;
  result.operator iter = new Iter_int() {
    int index = n;
    Iter_int retv;
    unravel retv;
    advance = new void() { index += skip; };
    get = new int() { return index; };
    if (skip > 0) {
      valid = new bool() { return index <= m; };
    } else if (skip < 0) {
      valid = new bool() { return index >= m; };
    } else {
      valid = new bool() { return false; };
    }
    return retv;
  };
  return result;
}

Iterable_int range(int n) {
  return range(0,n-1);
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
  return a+sequence(n+1)/n*(b-a);
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

string mapArrayString(string From, string To)
{
  return "typedef "+From+" From;
  typedef "+To+" To;
  To[] map(To f(From), From[] a) {
    return sequence(new To(int i) {return f(a[i]);},a.length);
  }";
}

// This function is deprecated: use
// from mapArray(Src=T1, Dst=T2) access map;
void mapArray(string From, string To)
{
  eval(mapArrayString(From,To),true);
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
  real clock;
}

struct cputime {
  processtime parent;
  processtime child;
  processtime change;
}

cputime cputime()
{
  static processtime last;
  real[] a=_cputime();
  cputime cputime;
  real clock=a[4];
  cputime.parent.user=a[0]; // Includes system time
  cputime.parent.system=0;
  cputime.parent.clock=clock;
  cputime.child.user=a[2];
  cputime.child.system=a[3];
  cputime.child.clock=0;
  real user=cputime.parent.user+cputime.child.user;
  real system=cputime.parent.system+cputime.child.system;
  cputime.change.user=user-last.user;
  cputime.change.system=system-last.system;
  cputime.change.clock=clock-last.clock;
  last.user=user;
  last.system=system;
  last.clock=clock;
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

struct realschur {
  real[][] U;
  real[][] T;
}

realschur schur(real[][] a)
{
  real[][][] S=_schur(a);
  realschur schur;
  schur.U=S[0];
  schur.T=S[1];
  return schur;
}

struct schur {
  pair[][] U;
  pair[][] T;
}

schur schur(pair[][] a)
{
  pair[][][] S=_schur(a);
  schur schur;
  schur.U=S[0];
  schur.T=S[1];
  return schur;
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

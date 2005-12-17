/*****
 * plain.asy
 * Andy Hammerlindl and John Bowman 2004/08/19
 *
 * A package for general purpose drawing, with automatic sizing of pictures.
 *
 *****/

include constants;

access version;		    
if(version.VERSION != VERSION()) {
  write(stdout,"Warning: using possibly incompatible version "+
	 version.VERSION+" of plain.asy"+'\n');
}
   
include transforms;
include pens;
include paths;
include filldraw;
include margins;
include picture;
include Label;
include arcs;
include boxes;
include markers;
include shipout;
include arrows;
include strings;

// Three-dimensional projections

typedef real[][] transform3;

struct projection {
  public triple camera;
  public transform3 project;
  public transform3 aspect;
  void init(triple camera, transform3 project, transform3 aspect) {
    this.camera=camera;
    this.project=project;
    this.aspect=aspect;
  }
  projection copy() {
    projection P=new projection;
    P.init(camera,project,aspect);
    return P;
  }
}

projection operator init() {return new projection;}
  
public projection currentprojection;

typedef void exitfcn();
void nullexitfcn();

void exitfunction()
{
  if(interact() || (!shipped && !currentpicture.empty())) shipout();
}
atexit(exitfunction);

// A restore thunk is a function, that when called, restores the graphics state
// to what it was when the restore thunk was created.
typedef void restoreThunk();

// When save is called, this will be redefined to do the corresponding restore.
void restore()
{
  write("warning: restore called with no matching save");
}

restoreThunk buildRestoreThunk()
{
  pen defaultpen=defaultpen();
  pen p=currentpen;
  picture pic=currentpicture.copy();
  projection P=currentprojection.copy();
  restoreThunk r=restore;
  return new void() {
    defaultpen(defaultpen);
    currentpen=p;
    currentpicture=pic;
    currentprojection=P;
    uptodate(false);
    restore=r;
  };
}

// Save the current state, so that restore will put things back in that state.
restoreThunk save() 
{
  return restore=buildRestoreThunk();
}

void restoredefaults()
{
  write("warning: restoredefaults called with no matching savedefaults");
}

restoreThunk buildRestoreDefaults()
{
  pen defaultpen=defaultpen();
  exitfcn atexit=atexit();
  restoreThunk r=restoredefaults;
  return new void() {
    defaultpen(defaultpen);
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
  atexit(nullexitfcn);
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
string[] reverse(string[] a) {return a[reverse(a.length)];}

void eval(string s, bool embedded=false)
{
  if(!embedded) initdefaults();
  _eval(s,embedded);
  if(!embedded) restoredefaults();
}

void eval(code s, bool embedded=false)
{
  if(!embedded) initdefaults();
  _eval(s,embedded);
  if(!embedded) restoredefaults();
}

void execute(string s, bool embedded=false)
{
  eval("include \""+s+"\";",embedded);
}

/*****
 * plain.asy
 * Andy Hammerlindl and John Bowman 2004/08/19
 *
 * A package for general purpose drawing, with automatic sizing of pictures.
 *
 *****/

access settings;
include constants;

access version;		    
if(version.VERSION != VERSION()) {
  write(stdout,"Warning: using possibly incompatible version "+
	 version.VERSION+" of plain.asy"+'\n');
}
   
include pens;
include paths;
include filldraw;
include margins;
include picture;
include shipout;
include Label;
include arcs;
include boxes;
include markers;
include arrows;
include strings;
include debugger;

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
typedef restoreThunk saveFunction();
saveFunction[] saveFunctions={};

// When save is called, this will be redefined to do the corresponding restore.
void restore()
{
  write("warning: restore called with no matching save");
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
        uptodate(false);
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
triple[] reverse(triple[] a) {return a[reverse(a.length)];}
string[] reverse(string[] a) {return a[reverse(a.length)];}

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

// Conditionally process each file name in array s in a new environment.
void asy(string format, bool overwrite=false ... string[] s)
{
  for(int i=0; i < s.length; ++i) {
    string f=s[i];
    int n=rfind(f,".asy");
    if(n != -1) f=erase(f,n,-1);
    if(overwrite || error(input(f+"."+format,check=false))) {
      string F="\""+f+"\"";
      string outname=settings.outname;
      string outformat=settings.outformat;
      bool interactiveView=settings.interactiveView;
      bool batchView=settings.batchView;
      settings.outname=f;
      settings.outformat=format;
      settings.interactiveView=false;
      settings.batchView=false;
      eval("import "+F+" as dummy; exitfunction()");
      settings.outname=outname;
      settings.outformat=outformat;
      settings.interactiveView=interactiveView;
      settings.batchView=batchView;
    }
  }
}

struct processtime {
  real user;
  real system;
}

processtime operator init() {return new processtime;}

struct cputime {
  processtime parent;
  processtime child;
}

cputime operator init() {return new cputime;}

cputime cputime() 
{
  real [] a=_cputime();
  cputime cputime;
  cputime.parent.user=a[0];
  cputime.parent.system=a[1];
  cputime.child.user=a[2];
  cputime.child.system=a[3];
  return cputime;
}

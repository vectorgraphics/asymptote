// Default file prefix used for inline LaTeX mode
string defaultfilename;

bool shipped; // Was a picture or frame already shipped out?

restricted bool Wait=true;                         
restricted bool NoWait=false;

frame patterns;

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame UpsideDown(frame f) {return rotate(180)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);
orientation orientation=Portrait;

void shipout(string prefix=defaultfilename,
             orientation orientation=orientation,
             string format="", bool wait=NoWait, bool view=true);

include plain_xasy;

struct GUIop
{
  transform[] Transform;
  bool[] Delete;
}

frame GUI[];
  
frame GUI(int index) {
  while(GUI.length <= index) {
    frame f;
    GUI.push(f);
  }
  return GUI[index];
}
                                                   
private struct DELETET {}
DELETET DELETE=null;

GUIop[] GUIlist;

// Delete item
void GUIop(int index, int filenum=0, DELETET)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop;
  GUIop GUIobj=GUIlist[filenum];
  while(GUIobj.Transform.length <= index) {
    GUIobj.Transform.push(identity());
    GUIobj.Delete.push(false);
  }
  GUIobj.Delete[index]=true;
}

// Transform item
void GUIop(int index, int filenum=0, transform T)
{
  if(GUIlist.length <= filenum) GUIlist[filenum]=new GUIop;
  GUIop GUIobj=GUIlist[filenum];
  while(GUIobj.Transform.length <= index) {
    GUIobj.Transform.push(identity());
    GUIobj.Delete.push(false);
  }
  GUIobj.Transform[index]=T*GUIobj.Transform[index];
}

private int GUIFilenum;

void GUIreset()
{
  GUIFilenum=0;
  GUI=new frame[];
  GUIlist=new GUIop[];
}

void shipout(string prefix=defaultfilename, frame f, frame preamble=patterns,
             string format="", bool wait=NoWait, bool view=true)
{
  if(inXasyMode)return;
  GUIreset();
  readGUI();
  bool Transform=GUIFilenum < GUIlist.length;
  if(GUI.length > 0) {
    frame F;
    add(F,f);
    for(int i=0; i < GUI.length; ++i)
      add(F,GUI(i));
    f=F;
  }
  
  // Applications like LaTeX cannot handle large PostScript coordinates.
  pair m=min(f);
  int limit=2000;
  if(abs(m.x) > limit || abs(m.y) > limit) f=shift(-m)*f;

  uptodate(true);

  int i=-1;
  transform[] t;
  bool[] d;
  GUIop GUIop;
  if(Transform) {
    GUIop=GUIlist[GUIFilenum];
    t=GUIop.Transform;
    d=GUIop.Delete;
  }
  //for the xformStack:
  shipout(prefix,f,preamble,format,wait,view,xformStack.pop);

  //for GUIop:
  //shipout(prefix,f,preamble,format,wait,view,
  //        new transform() {
  //        if(++i < d.length && d[i]) return (0,0,0,0,0,0);
  //        return i < t.length ? t[i] : identity();});

  shipped=true;
  ++GUIFilenum;
}

void shipout(string prefix=defaultfilename, picture pic,
             frame preamble=patterns, orientation orientation=orientation,
             string format="", bool wait=NoWait, bool view=true)
{
  shipout(prefix,orientation(pic.fit()),preamble,format,wait,view);
}

shipout=new void(string prefix=defaultfilename,
             orientation orientation=orientation,
             string format="", bool wait=NoWait, bool view=true) {
  uptodate(true);
  shipout(prefix,currentpicture,orientation,format,wait,view);
};

void newpage(picture pic=currentpicture) 
{
  tex(pic,"\newpage");
  layer(pic);
}

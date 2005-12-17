bool Wait=true;				
bool NoWait=false;

struct GUIop
{
  public transform[] Transform;
  public bool[] Delete;
}

GUIop operator init() {return new GUIop;}
  
frame GUI[];
  
frame GUI(int index) {
  while(GUI.length <= index) {
    frame f;
    GUI.push(f);
  }
  return GUI[index];
}
						   
private struct DELETET {}
public DELETET DELETE=null;

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
	     string format="", bool wait=NoWait, bool quiet=false)
{
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
  shipout(prefix,f,preamble,format,wait,quiet,
  	  Transform ? GUIlist[GUIFilenum].Transform : null,
	  Transform ? GUIlist[GUIFilenum].Delete : null);
  ++GUIFilenum;
  shipped=true;
  uptodate(true);
}

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);

void shipout(string prefix=defaultfilename, picture pic, real unitsize=0,
	     frame preamble=patterns, orientation orientation=Portrait,
	     string format="", bool wait=NoWait, bool quiet=false)
{
  shipout(prefix,
	  orientation(unitsize == 0 ? pic.fit() : pic.fit(scale(unitsize))),
	  preamble,format,wait,quiet);
}

void shipout(string prefix=defaultfilename, orientation orientation=Portrait,
	     real unitsize=0, string format="", bool wait=NoWait,
	     bool quiet=false)
{
  shipout(prefix,currentpicture,unitsize,orientation,format,wait,quiet);
}


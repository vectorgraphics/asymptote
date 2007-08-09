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

void shipout(string prefix=defaultfilename, frame f, frame preamble=patterns,
             string format="", bool wait=NoWait, bool view=true)
{
  if(inXasyMode) return;
  
  // Applications like LaTeX cannot handle large PostScript coordinates.
  pair m=min(f);
  int limit=2000;
  if(abs(m.x) > limit || abs(m.y) > limit) f=shift(-m)*f;

  uptodate(true);
  shipout(prefix,f,preamble,format,wait,view,xformStack.pop);
  shipped=true;
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

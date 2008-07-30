// Default file prefix used for inline LaTeX mode
string defaultfilename;

bool shipped; // Was a picture or frame already shipped out?

restricted bool Wait=true;                         
restricted bool NoWait=false;

frame currentpatterns;

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame UpsideDown(frame f) {return rotate(180)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);
orientation orientation=Portrait;

include plain_xasy;

object embed3(frame f);
object embed3(picture pic);

bool prc()
{
  return settings.prc && settings.outformat == "pdf";
}

frame enclose(object F)
{
  frame f;
  if(prc()) {
    frame out;
    label(out,F.L);
    f=out;
  } else f=F.f;
  return f;
}

void shipout(string prefix=defaultfilename, picture pic,
             orientation orientation=orientation,
             string format="", bool wait=NoWait, bool view=true);

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=NoWait, bool view=true)
{
  if(is3D(f))
    f=enclose(embed3(f));

  if(inXasyMode) {
    erase();
    add(f,group=false);
    return;
  }
  
  // Applications like LaTeX cannot handle large PostScript coordinates.
  pair m=min(f);
  int limit=2000;
  if(abs(m.x) > limit || abs(m.y) > limit) f=shift(-m)*f;

  shipout(prefix,f,currentpatterns,format,wait,view,xformStack.pop0);
  shipped=true;
}

shipout=new void(string prefix=defaultfilename, picture pic,
	orientation orientation=orientation,
	string format="", bool wait=NoWait, bool view=true)
{
  shipout(prefix,
	  orientation(pic.nodes3.length > 0 ? enclose(embed3(pic)) : pic.fit()),
	  format,wait,view);
};

void shipout(string prefix=defaultfilename,
             orientation orientation=orientation,
             string format="", bool wait=NoWait, bool view=true)
{
  shipout(prefix,currentpicture,orientation,format,wait,view);
}

void newpage(frame f)
{
  tex(f,"\newpage");
  layer(f);
}

void newpage(picture pic=currentpicture) 
{
  tex(pic,"\newpage");
  layer(pic);
}

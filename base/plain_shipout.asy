// Default file prefix used for inline LaTeX mode
string defaultfilename;

string outprefix(string prefix=defaultfilename) {
  string s=prefix != "" ? prefix :
    (settings.outname == "" && interactive()) ? "out" : settings.outname;
return stripdirectory(stripextension(s));
}

bool shipped; // Was a picture or frame already shipped out?

frame currentpatterns;

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame UpsideDown(frame f) {return rotate(180)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);
orientation orientation=Portrait;

object embed3(string, frame, string, string, projection);
string Embed(string name, string options="", real width=0, real height=0);
string Link(string label, string text, string options="");

bool prc0() {
  return (settings.prc && (settings.outformat == "pdf" || pdf())) ||
    settings.inlineimage;
}

bool prc() {
  return prc0() && Embed != null;
}

bool is3D()
{
  return prc() || settings.render != 0;
}

frame enclose(string prefix=defaultfilename, object F)
{
  if(prc()) {
    frame f;
    label(f,F.L);
    return f;
  } return F.f;
}

include plain_xasy;

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=false, bool view=true,
	     string options="", string script="",
	     projection P=currentprojection)
{
  if(is3D(f)) {
    f=enclose(prefix,embed3(prefix,f,options,script,P));
    if(settings.render != 0 && !prc()) {
      shipped=true;
      return;
    }
  }

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

void shipout(string prefix=defaultfilename, picture pic=currentpicture,
	     orientation orientation=orientation,
	     string format="", bool wait=false, bool view=true,
	     string options="", string script="",
	     projection P=currentprojection)
{
  if(!uptodate()) {
    bool inlinetex=settings.inlinetex;
    settings.inlinetex=settings.inlineimage;
    frame f=pic.fit(wait=wait,view=view,options,script,P);    
    if(!pic.empty2() || settings.render == 0 || prc())
      shipout(prefix,orientation(f),format,wait,view);
    settings.inlinetex=inlinetex;
  }
  
  pic.uptodate=true;
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

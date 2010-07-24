// Default file prefix used for inline LaTeX mode
string defaultfilename;

string[] file3;

string outprefix(string prefix=defaultfilename) {
  return stripextension(prefix != "" ? prefix : outname());
}

string outformat(string format="") 
{
  if(format == "") format=settings.outformat;
  if(format == "") format=nativeformat();
  return format;
}

bool shipped; // Was a picture or frame already shipped out?

frame currentpatterns;

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame UpsideDown(frame f) {return rotate(180)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);
orientation orientation=Portrait;

// Forward references to functions defined in module three.
object embed3(string, frame, string, string, string, light, projection);
string Embed(string name, string options="", real width=0, real height=0);
string Link(string label, string text, string options="");

bool prc0(string format="")
{
  return settings.prc && (outformat(format) == "pdf" || settings.inlineimage);
}

bool prc(string format="") {
  return prc0(format) && Embed != null;
}

bool is3D(string format="")
{
  return prc(format) || settings.render != 0;
}

frame enclose(string prefix=defaultfilename, object F, string format="")
{
  if(prc(format)) {
    frame f;
    label(f,F.L);
    return f;
  } return F.f;
}

include plain_xasy;

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=false, bool view=true,
	     string options="", string script="",
	     light light=currentlight, projection P=currentprojection)
{
  if(is3D(f)) {
    f=enclose(prefix,embed3(prefix,f,format,options,script,light,P));
    if(settings.render != 0 && !prc(format)) {
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
	     light light=currentlight, projection P=currentprojection)
{
  if(!uptodate()) {
    bool inlinetex=settings.inlinetex;
    bool prc=prc(format);
    bool empty3=pic.empty3();
    if(prc && !empty3) {
        if(settings.render == 0) {
        string image=outprefix(prefix)+"+"+(string) file3.length;
        if(settings.inlineimage) image += "_0";
        settings.inlinetex=false;
        settings.prc=false;
        shipout(image,pic,orientation,nativeformat(),view=false,light,P);
        settings.prc=true;
        settings.inlinetex=settings.inlineimage;
      }
    }
    frame f=pic.fit(prefix,format,view=view,options,script,light,P);
    if(!pic.empty2() || settings.render == 0 || prc || empty3)
      shipout(prefix,orientation(f),format,wait,view);
    settings.inlinetex=inlinetex;
  }
  
  pic.uptodate=true;
  shipped=true;
}

void newpage(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      newpage(f);
    },true);
}

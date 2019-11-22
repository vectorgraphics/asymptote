// Default file prefix used for inline LaTeX mode
string defaultfilename;

file _outpipe;
if(settings.xasy)
  _outpipe=output(mode="pipe");

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

frame currentpatterns;

frame Portrait(frame f) {return f;};
frame Landscape(frame f) {return rotate(90)*f;};
frame UpsideDown(frame f) {return rotate(180)*f;};
frame Seascape(frame f) {return rotate(-90)*f;};
typedef frame orientation(frame);
orientation orientation=Portrait;

// Forward references to functions defined in module three.
object embed3(string, frame, string, string, string, light, projection);
string Embed(string name, string text="", string options="", real width=0,
             real height=0);

bool prconly(string format="")
{
  return outformat(format) == "prc";
}

bool prc0(string format="")
{
  return settings.prc && (outformat(format) == "pdf" || prconly() || settings.inlineimage );
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

void deconstruct(picture pic=currentpicture)
{
  frame f;
  transform t=pic.calculateTransform();
  if(currentpicture.fitter == null)
    f=pic.fit(t);
  else
    f=pic.fit();
  deconstruct(f,currentpatterns,t);
}

bool implicitshipout=false;

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=false, bool view=true,
	     string options="", string script="",
	     light light=currentlight, projection P=currentprojection,
             transform t=identity)
{
  if(is3D(f)) {
    f=enclose(prefix,embed3(prefix,f,format,options,script,light,P));
    if(settings.render != 0 && !prc(format)) {
      return;
    }
  }

  if(outformat(format) == "html") {
    warning("htmltosvg",
            "html output requested for 2D picture; generating svg image instead...");
    format="svg";
  }
  
  if(settings.xasy || (!implicitshipout && prefix == defaultfilename)) {
    if(prefix == defaultfilename) {
      currentpicture.clear();
      add(f,group=false);
    }
    return;
  }
  
  // Applications like LaTeX cannot handle large PostScript coordinates.
  pair m=min(f);
  int limit=2000;
  if(abs(m.x) > limit || abs(m.y) > limit) f=shift(-m)*f;

  _shipout(prefix,f,currentpatterns,format,wait,view,t);
}

void shipout(string prefix=defaultfilename, picture pic=currentpicture,
	     orientation orientation=orientation,
	     string format="", bool wait=false, bool view=true,
	     string options="", string script="",
	     light light=currentlight, projection P=currentprojection)
{
  pic.uptodate=true;
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
      }
      settings.inlinetex=settings.inlineimage;
    }
    frame f;
    transform t=pic.calculateTransform();
    if(currentpicture.fitter == null)
      f=pic.fit(t);
    else
      f=pic.fit(prefix,format,view=view,options,script,light,P);

    if(!prconly() && (!pic.empty2() || settings.render == 0 || prc || empty3))
      shipout(prefix,orientation(f),format,wait,view,t);
    settings.inlinetex=inlinetex;
  }
}

void newpage(picture pic=currentpicture)
{
  pic.add(new void(frame f, transform) {
      newpage(f);
    },true);
}

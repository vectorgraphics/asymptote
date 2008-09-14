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

object embed3(string, frame, string, projection);
string embed(string name, string options="", real width=0, real height=0);
string link(string label, string text, string options="");

bool prc0() {
  return settings.prc && settings.outformat == "pdf";
}

bool prc() {
  return prc0() && embed != null;
}

bool is3D()
{
  return prc() || settings.render > 0;
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

frame psimage(string prefix=defaultfilename, bool view=true)
{
  string name=outprefix(prefix)+".ps";
  delete(name);
  string javascript="
console.println('Rasterizing to "+name+"');
var pp = this.getPrintParams();
pp.interactive = pp.constants.interactionLevel.silent;
pp.fileName = '"+name+"';
fv = pp.constants.flagValues;
pp.flags |= fv.suppressRotate;
pp.pageHandling = pp.constants.handling.none;
pp.printerName = 'FILE';
try{silentPrint(pp);} catch(e){this.print(pp);}";
  if(!view ||
     !(interactive() ? settings.interactiveView : settings.batchView))
    javascript += "this.closeDoc();";
  string s;
  if(pdf())
    s="\pdfannot width 1pt height 1pt { /AA << /PO << /S /JavaScript /JS ("+javascript+") >> >> }";
  else
    s="\special{ps: mark {Catalog} << /OpenAction << /S /JavaScript /JS ("+
      javascript+") >> >> /PUT pdfmark }";
  frame g;
  tex(g,s);
  return g;
}

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=false, bool view=true,
	     string options="", projection P=currentprojection)
{
  if(is3D(f)) {
    f=enclose(prefix,embed3(prefix,f,options,P));
    if(settings.render > 0 && !prc()) {
      shipped=true;
      return;
    }
  }

  if(settings.psimage && is3D())
    prepend(f,psimage(prefix,view));

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
	     string options="", projection P=currentprojection)
{
  if(!uptodate()) {
    frame f=pic.fit(wait=wait,view=view,options,P);
    if(currentpicture.nodes3.length == 0 || settings.render == 0 || prc())
      shipout(prefix,orientation(f),format,wait,view);
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

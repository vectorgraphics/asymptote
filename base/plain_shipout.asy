// Default file prefix used for inline LaTeX mode
string defaultfilename;

string outprefix(string prefix=defaultfilename) {
  return stripdirectory(stripextension(prefix == "" ? settings.outname :
				       prefix));
}

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

object embed3(string prefix, frame f);
object embed3(string prefix, picture pic);

bool prc()
{
  return settings.prc && settings.outformat == "pdf";
}

frame enclose(string prefix=defaultfilename, object F)
{
  if(prc()) {
    frame f;
    label(f,F.L);
    return f;
  } return F.f;
}

void shipout(string prefix=defaultfilename, frame f,
             string format="", bool wait=NoWait, bool view=true)
{
  if(is3D(f))
    f=enclose(prefix,embed3(prefix,f));

  if(settings.psimage && prc() && !pdf()) {
    string name=outprefix(prefix)+".ps";
    delete(name);
    string autoclose=view &&
      (interactive() ? settings.interactiveView : settings.batchView) ?
      "" : "this.closeDoc();";
    tex(f,"\special{ps: mark {Catalog} << /OpenAction << /S /JavaScript /JS (printToFile(\""+name+"\");"+autoclose+") >> >> /PUT pdfmark }");
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

void shipout(string prefix=defaultfilename, picture pic,
	orientation orientation=orientation,
	string format="", bool wait=NoWait, bool view=true)
{
  shipout(prefix,orientation(pic.nodes3.length > 0 ?
			     enclose(embed3(prefix,pic)) : pic.fit()),
	  format,wait,view);
}

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

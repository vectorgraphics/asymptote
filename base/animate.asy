/*****
 * animate.asy
 * Andy Hammerlindl 2005/11/06
 *
 * Produce animated gifs.
 *****/

// animation delay is in milliseconds
real animationdelay=50;

struct animation {
  static string outname() {
    return defaultfilename == "" ? settings.outname : defaultfilename;
  }

  picture[] pictures;
  string[] files;
  string prefix=outname();
  int index=0;

  static animation prefix(string s=outname()) {
    animation animation=new animation;
    animation.prefix=s;
    return animation;
  }
  
  private string nextname(string prefix=prefix) {
    string name=prefix+(string)index;
    ++index;
    return stripextension(stripdirectory(name));
  }

  void shipout(string prefix=prefix, picture pic=currentpicture) {
    string name=nextname(prefix);
    string format=nativeformat();
    shipout(name,pic,format=format,view=false);
    files.push(name+"."+format);
  }
  
  void add(picture pic=currentpicture) {
    pictures.push(pic.copy());
  }
  
  void purge(bool keep) {
    if(!(keep || settings.keep)) {
      for(int i=0; i < files.length; ++i)
        delete(files[i]);
    }
  }

  int merge(int loops=0, real delay=animationdelay, string format="gif",
            string options="", bool keep=false) {
    string args="-loop " +(string) loops+" -delay "+(string)(delay/10)+" "
      +options;
      for(int i=0; i < files.length; ++i)
        args += " " +files[i];
      int rc=convert(args,format=format);
      purge(keep);
      if(rc == 0) animate(format=format);
      else abort("merge failed");
      return rc;
  }

  pair min,max;

  // Export all frames with the same scaling.
  void export(string prefix=prefix, bool multipage=false) {
    if(pictures.length == 0) return;
    picture all;
    size(all,pictures[0]);
    for(int i=0; i < pictures.length; ++i) {
      add(all,pictures[i]);
      if(multipage) newpage(all);
    }
    if(multipage) {
      plain.shipout(prefix,all,view=false);
      return;
    }
    transform t=inverse(all.calculateTransform()*pictures[0].T);
    min=min(all);
    max=max(all);
    pair m=t*min;
    pair M=t*max;
    for(int i=0; i < pictures.length; ++i) {
      draw(pictures[i],m,nullpen);
      draw(pictures[i],M,nullpen);
      this.shipout(pictures[i]);
    }
  }
  bool pdflatex() {
    if(pdf() && latex()) return true;
    abort("error: PDF animations require -tex pdflatex");
    return false;
  }

  string pdf(real delay=animationdelay, string options="") {
    string filename="_"+stripextension(stripdirectory(prefix));
    if(!pdflatex()) return "";
    bool inlinetex=settings.inlinetex;
    settings.inlinetex=false;
    export(filename,true);
    settings.inlinetex=inlinetex;
    shipped=false;
    string s="\PDFAnimLoad[single,interval="+string(delay);
    if(options != "") s += ","+options;
    texpreamble(s+"]{"+prefix+"}{"+filename+"}{"+string(pictures.length)+"}%");

    if(!settings.keep && !settings.inlinetex) {
      exitfcn atexit=atexit();
      void exitfunction() {
        atexit();
        delete(filename+".pdf");
      }
      atexit(exitfunction);
    }

    return "\PDFAnimJSPageEnable\PDFAnimation{"+prefix+"}";
  }

  private string color(string name, pen p, bool colorspace=false) {
    string s='\\'+name+"{";
    real[] P=colors(p);
    if(P.length > 3) P=colors(rgb(p));
    for(int i=0; i < P.length; ++i)
      s += string(P[i])+" ";
    if(colorspace) {
      if(P.length == 1) s += "g";
      if(P.length == 3) s += "rg";
    }
    s += "}";
    return s;
  }

  string controlpanel(pen foreground=black, bool percentage=false,
                      pen background=green, pen border=invisible) {
    if(!pdflatex()) return "";
    string s="\PDFAnimButtons";
    if(percentage) s += "P";
    return s+"["+color("BG",background)+
      color("textColor",foreground,colorspace=true)+
      color("BC",border)+"]{"+prefix+"}";
  }

  private string field(string field, string max, pen foreground=black,
                       pen background=white, real margin=0) {
    if(!pdflatex()) return "";
    frame f;
    label(f,max,foreground);
    pair delta=max(f)-min(f);
    return '\\'+field+"["+color("BG",background)+
      color("textColor",foreground,colorspace=true)+
      "\textSize{"+string(fontsize(foreground))+"}]{"+prefix+"}{"+
      string(delta.x+labelmargin(foreground))+"bp}{"+
      string(delta.y+labelmargin(foreground)+margin)+"bp}";
  }

  string progress(pen foreground=black, pen background=white) {
    return field("PDFProgressField","100",foreground,background);
  }

  string delay(pen foreground=black, pen background=white) {
    return field("PDFAnimDelayButton","Delay",foreground,background,2);
  }

  int movie(int loops=0, real delay=animationdelay,
            string format=settings.outformat == "" ? "gif" : settings.outformat,
            string options="", bool keep=false) {
    export();
    return merge(loops,delay,format,options,keep);
  }

}

animation animation(string prefix) 
{
  return animation.prefix(prefix);
}

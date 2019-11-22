/*****
 * animation.asy
 * Andy Hammerlindl and John Bowman 2005/11/06
 *
 * Produce GIF, inline PDF, or other animations.
 *****/

// animation delay is in milliseconds
real animationdelay=50;

typedef frame enclosure(frame);

frame NoBox(frame f) {
  return f;
}

enclosure BBox(real xmargin=0, real ymargin=xmargin,
               pen p=currentpen, filltype filltype=NoFill) {
  return new frame(frame f) {
    box(f,xmargin,ymargin,p,filltype,above=false);
    return f;
  };
}

struct animation {
  picture[] pictures;
  string[] files;
  int index;

  string prefix;
  bool global; // If true, use a global scaling for all frames; this requires
  // extra memory since the actual shipout is deferred until all frames have
  // been generated. 

  void operator init(string prefix="", bool global=true) {
    prefix=replace(stripdirectory(outprefix(prefix))," ","_");
    this.prefix=prefix;
    this.global=global;
  }
  
  string basename(string prefix=stripextension(prefix)) {
    return "_"+prefix;
  }

  string name(string prefix, int index) {
    return stripextension(prefix)+"+"+string(index);
  }

  private string nextname() {
    string name=basename(name(prefix,index));
    ++index;
    return name;
  }

  void shipout(string name=nextname(), frame f) {
    string format=nativeformat();
    plain.shipout(name,f,format=format,view=false);
    files.push(name+"."+format);
  }
  
  void add(picture pic=currentpicture, enclosure enclosure=NoBox) {
    if(global) {
      ++index;
      pictures.push(pic.copy());
    } else this.shipout(enclosure(pic.fit()));
  }
  
  void purge(bool keep=settings.keep) {
    if(!keep) {
      for(int i=0; i < files.length; ++i)
        delete(files[i]);
    }
  }

  int merge(int loops=0, real delay=animationdelay, string format="gif",
            string options="", bool keep=settings.keep) {
    string args="-loop " +(string) loops+" -delay "+(string)(delay/10)+
      " -alpha Off -dispose Background "+options;
    for(int i=0; i < files.length; ++i)
      args += " " +files[i];
    int rc=convert(args,prefix+"."+format,format=format);
    this.purge(keep);
    if(rc == 0) animate(file=prefix+"."+format,format=format);
    else abort("merge failed");
    return rc;
  }

  void glmovie(string prefix=prefix, projection P=currentprojection) {
    if(!view() || settings.render == 0 || settings.outformat == "html") return;
    fit(prefix,pictures,view=true,P);
  }

  // Export all frames with the same scaling.
  void export(string prefix=prefix, enclosure enclosure=NoBox,
              bool multipage=false, bool view=false,
              projection P=currentprojection) {
    if(pictures.length == 0) return;
    if(!global) multipage=false;
    bool inlinetex=settings.inlinetex;
    if(multipage)
      settings.inlinetex=false;
    frame multi;
    frame[] fits=fit(prefix,pictures,view=false,P);
    for(int i=0; i < fits.length; ++i) {
      string s=name(prefix,i);
      if(multipage) {
        add(multi,enclosure(fits[i]));
        newpage(multi);
        files.push(s+"."+nativeformat());
      } else {
        if(pictures[i].empty3() || settings.render <= 0)
          this.shipout(s,enclosure(fits[i]));
        else // 3D frames
          files.push(s+"."+nativeformat());
      }
    }
    if(multipage) {
      plain.shipout(prefix,multi,view=view);
      settings.inlinetex=inlinetex;
    }
  }

  string load(int frames, real delay=animationdelay, string options="",
              bool multipage=false) {
    if(!global) multipage=false;
    string s="\animategraphics["+options+"]{"+format("%.18f",1000/delay,"C")+
      "}{"+basename();
    if(!multipage) s += "+";
    s += "}{0}{"+string(frames-1)+"}";
    return s;
  }

  bool pdflatex() 
  {
    return latex() && pdf();
  }

  string pdf(enclosure enclosure=NoBox, real delay=animationdelay,
             string options="", bool keep=settings.keep, bool multipage=true) {
    settings.twice=true;
    if(settings.inlinetex) multipage=true;
    if(!global) multipage=false;
    if(!pdflatex())
      abort("inline pdf animations require -tex pdflatex or -tex xelatex");
    if(settings.outformat != "") settings.outformat="pdf";
    
    string filename=basename();
    string pdfname=filename+".pdf";

    if(global)
      export(filename,enclosure,multipage=multipage);
    
    if(!keep) {
      exitfcn currentexitfunction=atexit();
      void exitfunction() {
        if(currentexitfunction != null) currentexitfunction();
        if(multipage || !settings.inlinetex)
          this.purge();
        if(multipage && !settings.inlinetex)
          delete(pdfname);
      }
      atexit(exitfunction);
    }

    if(!multipage)
      delete(pdfname);

    return load(index,delay,options,multipage);
  }

  int movie(enclosure enclosure=NoBox, int loops=0, real delay=animationdelay,
            string format=settings.outformat == "" ? "gif" : settings.outformat,
            string options="", bool keep=settings.keep) {
    if(global) {
      if(format == "pdf") {
        export(enclosure,multipage=true,view=true);
        return 0;
      }
      export(enclosure);
    }
    return merge(loops,delay,format,options,keep);
  }
}

animation operator init() {
  animation a=animation();
  return a;
}

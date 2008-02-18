/*****
 * animation.asy
 * Andy Hammerlindl and John Bowman 2005/11/06
 *
 * Produce animated gifs.
 *****/

// animation delay is in milliseconds
real animationdelay=50;

typedef frame fit(picture);

frame NoBox(picture pic) {
  return pic.fit();
}

fit BBox(real xmargin=0, real ymargin=xmargin,
	 pen p=currentpen, filltype filltype=NoFill) {
  return new frame(picture pic) {
    return bbox(pic,xmargin,ymargin,p,filltype);
  };
}

struct animation {
  string outname() {
    return "_"+(defaultfilename == "" ? settings.outname : defaultfilename);
  }

  picture[] pictures;
  string[] files;
  int index;

  string prefix;
  bool global; // If true, use a global scaling for all frames; this requires
  // extra memory since the actual shipout is deferred until all frames have
  // been generated. 

  void operator init(string s=outname(), bool global=true) {
    this.prefix=s;
    this.global=global;
  }
  
  string name(string prefix, int index) {
    return stripextension(stripdirectory(prefix+string(index)));
  }

  private string nextname(string prefix) {
    string name=name(prefix,index);
    ++index;
    return name;
  }

  void shipout(string prefix=prefix, string name=nextname(prefix), frame f) {
    string format=nativeformat();
    shipout(name,f,format=format,view=false);
    files.push(name+"."+format);
    shipped=false;
  }
  
  string pdfname() {
    return stripextension(stripdirectory(prefix));
  }

  void add(picture pic=currentpicture) {
    if(global) {
      pictures.push(pic.copy());
    } else this.shipout(pdfname(),pic.fit());
  }
  
  void purge(bool keep=settings.keep) {
    if(!keep) {
      for(int i=0; i < files.length; ++i)
        delete(files[i]);
    }
  }

  int merge(int loops=0, real delay=animationdelay, string format="gif",
            string options="", bool keep=settings.keep) {
    string args="-loop " +(string) loops+" -delay "+(string)(delay/10)+" "
      +options;
      for(int i=0; i < files.length; ++i)
        args += " " +files[i];
      int rc=convert(args,format=format);
      this.purge(keep);
      if(rc == 0) animate(format=format);
      else abort("merge failed");
      return rc;
  }

  // Export all frames with the same scaling.
  void export(string prefix=prefix, fit fit=NoBox,
	      bool multipage=false, bool view=false) {
    if(pictures.length == 0) return;
    picture all;
    size(all,pictures[0]);
    for(int i=0; i < pictures.length; ++i)
      add(all,pictures[i]);
    transform t=inverse(all.calculateTransform()*pictures[0].T);
    pair m=t*min(all);
    pair M=t*max(all);
    frame multi;
    for(int i=0; i < pictures.length; ++i) {
      draw(pictures[i],m,nullpen);
      draw(pictures[i],M,nullpen);
      if(multipage) {
	add(multi,fit(pictures[i]));
	newpage(multi);
      } else
	this.shipout(prefix,name(prefix,i),fit(pictures[i]));
    }
    if(multipage) {
      bool inlinetex=settings.inlinetex;
      settings.inlinetex=false;
      plain.shipout(prefix,multi,view=view);
      settings.inlinetex=inlinetex;
    }
    shipped=true;
  }

  string load(int frames, real delay=animationdelay, string options="") {
    return "\animategraphics["+options+"]{"+string(1000/delay)+"}{"+
      pdfname()+"}{0}{"+string(frames-1)+"}";
  }

  string pdf(fit fit=NoBox, real delay=animationdelay, string options="",
             bool keep=false, bool multipage=true) {
    if(settings.tex != "pdflatex")
      abort("inline pdf animations require -tex pdflatex");
    
    string filename=pdfname();
    bool single=global && multipage;

    if(global)
      export(filename,fit,multipage=multipage);
    shipped=false;

    if(!settings.keep && !settings.inlinetex) {
      exitfcn atexit=atexit();
      void exitfunction() {
        atexit();
        this.purge();
        if(!keep && single)
          delete(pdfname()+".pdf");
      }
      atexit(exitfunction);
    }

    if(!single)
      delete(filename+".pdf");

    return load(pictures.length,delay,options);
  }

  int movie(fit fit=NoBox, int loops=0, real delay=animationdelay,
            string format=settings.outformat == "" ? "gif" : settings.outformat,
            string options="", bool keep=false) {
    if(global && format == "pdf") {
      export(settings.outname,fit,multipage=true,view=true);
      return 0;
    }

    if(global)
      export(fit);
    return merge(loops,delay,format,options,keep);
  }
}

animation operator init() {
  animation a=animation();
  return a;
}

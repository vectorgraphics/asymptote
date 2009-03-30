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
  picture[] pictures;
  string[] files;
  int index;

  string prefix;
  bool global; // If true, use a global scaling for all frames; this requires
  // extra memory since the actual shipout is deferred until all frames have
  // been generated. 

  void operator init(string prefix="", bool global=true) {
    prefix=(prefix == "") ? outprefix() : stripdirectory(prefix);
    this.prefix=prefix;
    this.global=global;
  }
  
  string basename(string prefix=prefix) {
    return "_"+stripextension(prefix);
  }

  string name(string prefix, int index) {
    return stripextension(prefix)+string(index);
  }

  private string nextname() {
    string name=name(prefix,index);
    ++index;
    return name;
  }

  void shipout(string name=nextname(), frame f) {
    string format=nativeformat();
    plain.shipout(name,f,format=format,view=false);
    files.push(name+"."+format);
    shipped=false;
  }
  
  void add(picture pic=currentpicture, fit fit=NoBox) {
    if(global) {
      pictures.push(pic.copy());
    } else this.shipout(nextname(),fit(pic));
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
    bool inlinetex=settings.inlinetex;
    if(multipage)
      settings.inlinetex=false;
    for(int i=0; i < pictures.length; ++i) {
      draw(pictures[i],m,nullpen);
      draw(pictures[i],M,nullpen);
      if(multipage) {
        add(multi,fit(pictures[i]));
        newpage(multi);
      } else {
        if(pictures[i].empty3() || settings.render <= 0) {
          real render=settings.render;
          settings.render=0;
          this.shipout(name(prefix,i),fit(pictures[i]));
          settings.render=render;
        } else { // Render 3D frames
          string name=defaultfilename;
          defaultfilename=name(prefix,i);
          files.push(defaultfilename+"."+nativeformat());
          fit(pictures[i]);
          defaultfilename=name;
        }
      }
    }
    if(multipage) {
      plain.shipout(prefix,multi,view=view);
      settings.inlinetex=inlinetex;
    }
    shipped=true;
  }

  string load(int frames, real delay=animationdelay, string options="") {
    return "\animategraphics["+options+"]{"+format("%.18f",1000/delay,"C")+"}{"+
      basename()+"}{0}{"+string(frames-1)+"}";
  }

  string pdf(fit fit=NoBox, real delay=animationdelay, string options="",
             bool keep=settings.keep, bool multipage=true) {
    if(settings.inlinetex) multipage=true;
    if(settings.tex != "pdflatex")
      abort("inline pdf animations require -tex pdflatex");
    
    string filename=basename();
    string pdfname=filename+".pdf";
    bool single=global && multipage;

    if(global)
      export(filename,fit,multipage=multipage);
    
    shipped=false;

    if(!keep && !settings.inlinetex) {
      exitfcn currentexitfunction=atexit();
      void exitfunction() {
        if(currentexitfunction != null) currentexitfunction();
        this.purge();
        if(single)
          delete(pdfname);
      }
      atexit(exitfunction);
    }

    if(!single)
      delete(pdfname);

    return load(pictures.length,delay,options);
  }

  int movie(fit fit=NoBox, int loops=0, real delay=animationdelay,
            string format=settings.outformat == "" ? "gif" : settings.outformat,
            string options="", bool keep=settings.keep) {
    if(global && format == "pdf") {
      export(fit,multipage=true,view=true);
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

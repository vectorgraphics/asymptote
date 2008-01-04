/*****
 * animate.asy
 * Andy Hammerlindl 2005/11/06
 *
 * Produce animated gifs.
 *****/

// animation delay is in milliseconds
real animationdelay=50;

usepackage("animate");

struct animation {
  string outname() {
    return "_"+(defaultfilename == "" ? settings.outname : defaultfilename);
  }

  picture[] pictures;
  string[] files;
  int index;

  string prefix;
  bool global; // Use a global scaling for all frames; this requires
  // extra memory since the actual shipout is deferred until all frames have
  // been generated. 

  void operator init(string s=outname(), bool global=true) {
    this.prefix=s;
    this.global=global;
  }
  
  private string nextname(string prefix=prefix) {
    string name=prefix+string(index);
    ++index;
    return stripextension(stripdirectory(name));
  }

  void shipout(string prefix=prefix, picture pic=currentpicture) {
    string name=nextname(prefix);
    string format=nativeformat();
    shipout(name,pic,format=format,view=false);
    files.push(name+"."+format);
    shipped=false;
  }
  
  string pdfname() {
    return stripextension(stripdirectory(prefix));
  }

  void add(picture pic=currentpicture) {
    if(global) {
      pictures.push(pic.copy());
      index=pictures.length;
    } else this.shipout(pdfname(),pic);
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
      bool inlinetex=settings.inlinetex;
      settings.inlinetex=false;
      plain.shipout(prefix,all,view=false);
      settings.inlinetex=inlinetex;
      shipped=false;
      return;
    }
    index=0;
    transform t=inverse(all.calculateTransform()*pictures[0].T);
    min=min(all);
    max=max(all);
    pair m=t*min;
    pair M=t*max;
    for(int i=0; i < pictures.length; ++i) {
      draw(pictures[i],m,nullpen);
      draw(pictures[i],M,nullpen);
      this.shipout(prefix,pictures[i]);
    }
  }

  string load(int frames, real delay=animationdelay, string options="") {
    return "\animategraphics["+options+"]{"+string(1000/delay)+"}{"+
      pdfname()+"}{0}{"+string(frames-1)+"}";
  }

  string pdf(real delay=animationdelay, string options="",
	     bool multipage=true) {
    string filename=pdfname();

    if(global)
      export(filename,multipage=multipage);

    if(!settings.keep && !settings.inlinetex) {
      exitfcn atexit=atexit();
      void exitfunction() {
	atexit();
	this.purge();
      }
      atexit(exitfunction);
    }

    if(!global || !multipage)
      delete(filename+".pdf");

    return load(index,delay,options);
  }

  int movie(int loops=0, real delay=animationdelay,
            string format=settings.outformat == "" ? "gif" : settings.outformat,
            string options="", bool keep=false) {
    if(global)
      export();
    shipped=true;
    return merge(loops,delay,format,options,keep);
  }

}

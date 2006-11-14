/*****
 * animate.asy
 * Andy Hammerlindl 2005/11/06
 *
 * Produce animated gifs.
 *****/

struct animation {
  picture[] pictures;
  string[] files;
  string prefix=settings.outname;
  int index=0;

  static animation prefix(string s=settings.outname) {
    animation animation=new animation;
    animation.prefix=s;
    return animation;
  }
  
  private string nextname() {
    string name=prefix+(string)index;
    ++index;
    return name;
  }

  void shipout(picture pic=currentpicture) {
    string name=nextname();
    string format="eps";
    shipout(name,pic,format=format,view=false);
    files.push(name+"."+format);
  }
  
  void add(picture pic=currentpicture) {
    pictures.push(pic.copy());
  }
  
  int merge(int loops=0, int delay=50, string format="gif",
	    string options="", bool keep=false) {
    bool Keep=settings.keep;
    string Outname=settings.outname;
    settings.keep=keep;
    settings.outname=prefix;
    int rc=merge(files,"-loop " +(string) loops+" -delay "+(string)delay+
                 " "+options,format);
    settings.keep=Keep;
    settings.outname=Outname;
    return rc;
  }

  // delay is in units of 0.01s
  int movie(int loops=0, int delay=50, string format="gif",
	    string options="", bool keep=false) {
    if(pictures.length == 0) return 0;
    picture all;
    size(all,pictures[0].xsize,pictures[0].ysize,pictures[0].keepAspect);
    unitsize(all,pictures[0].xunitsize,pictures[0].yunitsize);
    for(int i=0; i < pictures.length; ++i)
      add(all,pictures[i]);
    pair m=truepoint(all,SW);
    pair M=truepoint(all,NE);
    for(int i=0; i < pictures.length; ++i) {
      picture pic=pictures[i];
      draw(pic,m,invisible);
      draw(pic,M,invisible);
      this.shipout(pic);
    }
    return merge(loops,delay,format,options,keep);
  }
}

animation operator init() {return new animation;}

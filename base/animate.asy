/*****
 * animate.asy
 * Andy Hammerlindl 2005/11/06
 *
 * Produce animated gifs.
 *****/

access settings;

struct animation {
  string prefix=settings.outname;
  int index=0;
  string[] files;

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

  void shipout(frame f) {
    string name=nextname();
    string format="eps";
    shipout(name,f,format=format,view=false);
    files.push(name+"."+format);
  }

  void shipout(picture pic=currentpicture) {
    this.shipout(pic.fit());
  }
  
  // delay is in units of 0.01s
  int merge(int loops=0, int delay=50, string format="gif",
            string options="", bool keep=false) {
    bool Keep=settings.keep;
    string Outname=settings.outname;
    settings.keep=keep;
    settings.outname=prefix;
    int rc=merge(files,"-loop " +(string) loops+" -delay "+(string)delay
                 +" "+options,format);
    settings.keep=Keep;
    settings.outname=Outname;
    return rc;
  }
}

animation operator init() {return new animation;}

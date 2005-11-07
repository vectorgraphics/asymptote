/*****
 * animate.asy
 * Andy Hammerlindl 2005/11/06
 *
 * Produce animated gifs.
 *****/

struct animation {
  string prefix=fileprefix();
  int index=0;
  string[] files;

  private string nextname() {
    string name=prefix+(string)index;
    ++index;
    return name;
  }

  void shipout(frame f) {
    string name=nextname();
    string format="eps";
    shipout(name,f,format=format,quiet=true);
    files.push(name+"."+format);
  }

  void shipout(picture pic=currentpicture) {
    this.shipout(pic.fit());
  }
  
  // delay is in units of 0.01s
  int merge(int loops=0, int delay=50, string format="gif", bool keep=false) {
    return merge(files,"-loop " +(string) loops+" -delay "+(string)delay,
		 format,keep);
  }
}

animation operator init() {return new animation;}

static public pen textpen=basealign;
static public pair align=1e-10*NE; 

// These compatibility routines for the pstoedit backend do not clip
// picture size data (pstoedit doesn't use automatic sizing). 
void beginclip(picture pic=currentpicture, path g, pen p=currentpen)
{
  pic.clippingwarning();
  pic.add(new void (frame f, transform t) {
    beginclip(f,t*g,p);
  });
}

void beginclip(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  pic.clippingwarning();
  pic.add(new void (frame f, transform t) {
    beginclip(f,t*g,p);
  });
}

void endclip(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    endclip(f);
  });
}

void gsave(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    gsave(f);
  });
}

void grestore(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    grestore(f);
  });
}

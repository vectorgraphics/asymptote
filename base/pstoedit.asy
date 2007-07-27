pen textpen=basealign;
pair align=1e-10*NE; 

// Compatibility routines for the pstoedit (version 3.43 or later) backend.
// These do not clip picture size data (pstoedit doesn't use automatic sizing). 
void beginclip(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  pic.add(new void (frame f, transform t) {
      beginclip(f,t*g,p);
    },true);
}

void endclip(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
      endclip(f);
    },true);
}

void gsave(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
      if(settings.deconstruct != 0) gsave(f);
    },true);
}

void grestore(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
      if(settings.deconstruct != 0) grestore(f);
    },true);
}
    

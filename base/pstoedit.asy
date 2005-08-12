static public pen textpen=basealign;
static public pair align=1e-10*NE; 

// Compatibility routines for the pstoedit (version 3.41 or later) backend.
// Note: apply the patch 
// These do not clip picture size data (pstoedit doesn't use automatic sizing). 
void beginclip(picture pic=currentpicture, path[] g, pen p=currentpen)
{
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
    if(!deconstruct()) gsave(f);
  });
}

void grestore(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
    if(!deconstruct()) grestore(f);
  });
}

void label(string s, real angle=0, pair position,
	   pair align=0, pen p=currentpen)
{
  (rotate(angle)*Label(s,position,align,p)).out();
}
    

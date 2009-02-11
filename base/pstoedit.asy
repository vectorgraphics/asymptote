pen textpen=basealign;
pair align=1e-10*NE; 

// Compatibility routines for the pstoedit (version 3.43 or later) backend.
void gsave(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
      gsave(f);
    },true);
}

void grestore(picture pic=currentpicture)
{
  pic.add(new void (frame f, transform) {
      grestore(f);
    },true);
}
    

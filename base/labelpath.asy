usepackage("pstricks");
usepackage("pst-text");

string LeftJustified="l";
string RightJustified="r";
string Centered="c";

void labelpath(frame f, Label L, path g, string justify=Centered,
               pen p=currentpen)
{
  if(latex() && !pdf()) {
    _labelpath(f,L.s,L.size,g,justify,(L.T.x,L.T.y+0.5linewidth(p)),p);
    return;
  }
  warning("labelpathlatex","labelpath requires -tex latex");
}

void labelpath(picture pic=currentpicture, Label L, path g,
               string justify=Centered, pen p=currentpen)
{
  pic.add(new void(frame f, transform t) {
      labelpath(f,L,t*g,justify,p);
    });
  frame f;
  label(f,Label(L.s,L.size));
  real w=size(f).y+L.T.y+0.5linewidth(p);
  pic.addBox(min(g),max(g),-w,w);
}

// MetaPost compatibility routines

public pen background=white;
public path cuttings;

path cutbefore(path p, path q) 
{
  slice s=firstcut(p,q);
  cuttings=s.before;
  return s.after;
}

path cutafter(path p, path q) 
{
  slice s=lastcut(p,q);
  cuttings=s.after;
  return s.before;
}

void unfill(picture pic=currentpicture, path g) 
{
  fill(pic,g,background);
}

void unfilldraw(picture pic=currentpicture, path g) 
{
  filldraw(pic,g,background);
}


// MetaPost compatibility routines

public pen background=white;

path cutbefore(path p, path q) 
{
  return firstcut(p,q).after;
}

path cutafter(path p, path q) 
{
  return lastcut(p,q).before;
}

void unfill(picture pic=currentpicture, path g) 
{
  fill(pic,g,background);
}

void unfilldraw(picture pic=currentpicture, path g) 
{
  filldraw(pic,g,background);
}


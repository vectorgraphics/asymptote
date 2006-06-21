// MetaPost compatibility routines

path cuttings;

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



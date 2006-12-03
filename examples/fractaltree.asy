size(200);

path ltrans(path p,int d)
{
  path a=rotate(65)*scale(0.4)*p;
  return shift(point(p,(1/d)*length(p))-point(a,0))*a;
}
path rtrans(path p, int d)
{
  path a=reflect(point(p,0),point(p,length(p)))*rotate(65)*scale(0.35)*p;
  return shift(point(p,(1/d)*length(p))-point(a,0))*a;
}

void drawtree(int depth, path branch)
{
  if(depth == 0) return;
  real breakp=(1/depth)*length(branch);
  draw(subpath(branch,0,breakp),deepgreen);
  drawtree(depth-1,subpath(branch,breakp,length(branch)));
  drawtree(depth-1,ltrans(branch,depth));
  drawtree(depth-1,rtrans(branch,depth));
  return;
}

path start=(0,0)..controls (-1/10,1/3) and (-1/20,2/3)..(1/20,1);
drawtree(6,start);


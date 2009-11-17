import graph;

size(100);

int n=30;
real minRadius=0.2;
real angles[]=uniform(0,2pi,n);
angles.delete(angles.length-1);

real[] r=new real[n];
for(int i=0; i < n; ++i)
  r[i]=unitrand()*(1-minRadius)+minRadius;

interpolate join=operator ..(operator tension(10,true));
draw(join(polargraph(r,angles,join),cycle),dot(red));



import graph;
import stats;

size(400,200,IgnoreAspect);

int n=10000;
real[] a=new real[n];
for(int i=0; i < n; ++i) a[i]=Gaussrand();

int nbins=100;
real dx=(max(a)-min(a))/(nbins-1);
real[] x=min(a)-dx/2+sequence(nbins+1)*dx;
real[] freq=frequency(a,x);
freq /= (dx*sum(freq));
histogram(freq,x);

draw(graph(Gaussian,min(a),max(a)),red);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$dP/dx$",LeftRight,RightTicks);



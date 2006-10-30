import graph;
import stats;

size(400,200,IgnoreAspect);

int n=10000;
real[] a=new real[n];
for(int i=0; i < n; ++i) a[i]=Gaussrand();

histogram(a,min(a),max(a),n=100,normalize=true,low=0);

draw(graph(Gaussian,min(a),max(a)),red);

xaxis("$x$",BottomTop,LeftTicks);
yaxis("$dP/dx$",LeftRight,RightTicks);



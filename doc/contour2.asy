import contour;
size(200);

real f(real x, real y) {return x^2-y^2;}
int n=25;
real[] c = new real[n];
for(int i=0; i < n; ++i) c[i]=(i-n/2)/n;

pen[] p=sequence(new pen(int i) {return c[i] >= 0 ? solid : dashed;},n);

draw(contour(f,(-1,-1),(1,1),c,50),p);

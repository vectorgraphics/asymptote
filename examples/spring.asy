pair coilpoint(real lambda, real r, real t)
{
  return (2.0*lambda*t+r*cos(t),r*sin(t));
}
  
guide coil(guide g=nullpath, real lambda, real r, real a, real b, int n)
{
  real width=(b-a)/n;
  for(int i=0; i <= n; ++i) {
    real t=a+width*i;
    g=g..coilpoint(lambda,r,t);
  }
  return g;
}

void drawspring(real x, string label) {
  real r=8;
  real t1=-pi; 
  real t2=10*pi;
  real lambda=(t2-t1+x)/(t2-t1);
  pair b=coilpoint(lambda,r,t1);
  pair c=coilpoint(lambda,r,t2);
  pair a=b-20;
  pair d=c+20;
 
  draw(a--b,BeginBar(2*barsize()));
  draw(c--d);
  draw(coil(lambda,r,t1,t2,100));
  dot(d);

  pair h=20*I;
  draw(label,a-h--d-h,red,Arrow,Bars,PenMargin);
}

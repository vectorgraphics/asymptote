unitsize(2cm);
import graph;
pair F(real t) {
  return (1.3*t,-4.5*t^2+3.0*t+1.0);
}
pair Fprime(real t) {
  return (1.3,-9.0*t+3.0);
}
path g=graphwithderiv(F,Fprime,0,0.9,4);
dot(g,red);
draw(g,arrow=Arrow(TeXHead));

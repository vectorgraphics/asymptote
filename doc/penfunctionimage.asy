import palette;

size(200);

real fracpart(real x) {return (x-floor(x));}

pair pws(pair z) {
  pair w=(z+exp(pi*I/5)/0.9)/(1+z/0.9*exp(-pi*I/5));
  return exp(w)*(w^3-0.5*I);
}

int N=512;

pair a=(-1,-1);
pair b=(0.5,0.5);
real dx=(b-a).x/N;
real dy=(b-a).y/N;

pen f(int u, int v) {
  pair z=a+(u*dx,v*dy);
  pair w=pws(z);
  real phase=degrees(w,warn=false);
  real modulus=w == 0 ? 0: fracpart(log(abs(w)));
  return hsv(phase,1,sqrt(modulus));
}

image(f,N,N,(0,0),(300,300),antialias=true);

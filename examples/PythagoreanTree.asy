size(250);

real a=3;
real b=4;
real c=hypot(a,b);

transform ta=shift(c,c)*rotate(-aCos(a/c))*scale(a/c)*shift(-c);
transform tb=shift(0,c)*rotate(aCos(b/c))*scale(b/c);

picture Pythagorean(int n) {
  picture pic;
  fill(pic,scale(c)*unitsquare,1/(n+1)*green+n/(n+1)*brown);
  if(n == 0) return pic;
  picture branch=Pythagorean(--n);
  add(pic,ta*branch);
  add(pic,tb*branch);
  return pic;
}

add(Pythagorean(12));

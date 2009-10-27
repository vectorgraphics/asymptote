path g=scale(100)*unitcircle;
pen p=linewidth(1cm);

frame f;
// Equivalent to draw(f,g,p):
fill(f,strokepath(g,p),red);
shipout("strokepathframe",f);
shipped=false;

size(400);

// Equivalent to draw(g,p):
add(new void(frame f, transform t) {
    fill(f,strokepath(t*g,p),red);
  });
currentpicture.addPath(g,p);


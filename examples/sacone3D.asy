import solids;

size(0,75);
real r=1;
real h=1;

revolution R=cone(r,h);

draw(surface(R),lightgreen+opacity(0.5),render(compression=Low));
pen edge=blue+0.25mm;
draw("$\ell$",(0,r,0)--(0,0,h),W,edge);
draw("$r$",(0,0,0)--(r,0,0),red+dashed);
draw((0,0,0)--(0,0,h),red+dashed);
dot(h*Z);

size(0,100);
import solids;
currentlight=Viewport;

triple v=O;
real r=1;
real h=1.5;
triple axis=Y+Z;

// Optimized cylinder
surface cylinder=shift(v)*align(unit(axis))*scale(r,r,h)*unitcylinder;
draw(cylinder,green,render(merge=true));

// Skeleton
revolution r=cylinder(v,r,h,axis);
//draw(surface(r),green,render(merge=true));
draw(r,blue+0.15mm);

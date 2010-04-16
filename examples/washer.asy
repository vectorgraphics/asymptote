import three;
size(10cm);

path3[] p=reverse(unitcircle3)^^scale3(0.5)*unitcircle3;
path[] g=reverse(unitcircle)^^scale(0.5)*unitcircle;
triple H=-0.4Z;

draw(surface(p,planar=true));
draw(surface(shift(H)*p,planar=true));
material m=material(lightgray,shininess=1.0);
for(path pp : g)
  draw(extrude(pp,H),m);


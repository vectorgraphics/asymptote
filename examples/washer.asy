import three;
size(10cm);

path3[] p=reverse(unitcircle3)^^scale3(0.5)*unitcircle3;
path[] g=reverse(unitcircle)^^scale(0.5)*unitcircle;
triple H=-0.4Z;

render render=render(merge=true);
draw(surface(p,planar=true),render);
draw(surface(shift(H)*p,planar=true),render);
material m=material(lightgray,shininess=1.0);
for(path pp : g)
  draw(extrude(pp,H),m);


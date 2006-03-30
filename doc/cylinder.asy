import solids;

size(0,100);

guide3[] g=cylinder(circle(O,1,Z),1.5,Z+Y);
draw(g);
triple M=max(g);

xaxis(Label("$x$",1),O,M.x,red,Below);
yaxis(Label("$y$",1),O,M.y,red,Below);
zaxis(Label("$z$",1),O,M.z,red,Below);

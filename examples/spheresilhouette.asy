import solids;
settings.render=0;
settings.prc=false;

size(200);

revolution r=sphere(O,1);
skeleton s;
r.transverse(s,0.5*length(r.g));
draw(s.transverse.front);
draw(s.transverse.back,dashed);
draw(r.silhouette());


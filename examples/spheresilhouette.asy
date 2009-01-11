import solids;
settings.render=0;
settings.prc=false;

size(200);

revolution r=sphere(O,1);
draw(r,1,longitudinalpen=nullpen);
draw(r.silhouette());

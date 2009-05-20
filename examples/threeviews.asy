import three;

picture pic;
size(pic,200);
currentlight.viewport=false;
settings.render=4;

draw(pic,scale3(0.5)*unitsphere,green);
draw(pic,Label("$x$",1),O--X);
draw(pic,Label("$y$",1),O--Y);
draw(pic,Label("$z$",1),O--Z);

frame Front=pic.fit(orthographic(-Y));
add(Front);
frame Top=pic.fit(orthographic(Z));
add(shift(0,min(Front).y-max(Top).y)*Top);
frame Right=pic.fit(orthographic(X));
add(shift(min(Front).x-max(Right).x)*Right);

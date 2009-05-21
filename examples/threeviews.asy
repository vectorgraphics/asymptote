import three;

picture pic;
size(pic,200);
currentlight.viewport=false;
settings.render=4;

draw(pic,scale3(0.5)*unitsphere,green);
draw(pic,Label("$x$",1),O--X);
draw(pic,Label("$y$",1),O--Y);
draw(pic,Label("$z$",1),O--Z);

addViews(pic);

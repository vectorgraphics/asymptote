import three;

picture pic;
unitsize(pic,5cm);

if(settings.render < 0) settings.render=4;
settings.toolbar=false;
viewportmargin=(1cm,1cm);

draw(pic,scale3(0.5)*unitsphere,green,render(compression=Low,merge=true));

draw(pic,Label("$x$",1),O--X);
draw(pic,Label("$y$",1),O--Y);
draw(pic,Label("$z$",1),O--Z);

// Europe and Asia:
//addViews(pic,ThreeViewsFR);
//addViews(pic,SixViewsFR);

// United Kingdom, United States, Canada, and Australia:
addViews(pic,ThreeViewsUS);
//addViews(pic,SixViewsUS);

// Front, Top, Right,
// Back, Bottom, Left:
//addViews(pic,SixViews);

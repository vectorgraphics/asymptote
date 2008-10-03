import graph3;

size(0,200);
size3(200,IgnoreAspect);

currentprojection=perspective(5,2,2);

scale(Linear,Linear,Log);

xaxis3("$x$",0,1,red,OutTicks(2,2));
yaxis3("$y$",0,1,red,OutTicks(2,2));
zaxis3("$z$",1,30,red,OutTicks(beginlabel=false));

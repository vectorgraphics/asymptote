import three;

currentlight=Headlamp;
size(469.75499pt,0);

currentprojection=perspective(
camera=(160.119024441391,136.348802919248,253.822628496226),
up=(-0.188035408976828,0.910392236102215,-0.368549401594584),
target=(25.5462739598034,1.77605243766079,-9.93996244768584),
zoom=5.59734733413271,
angle=5.14449021168139,
viewportshift=(0.813449720559684,-0.604674743165144),
autoadjust=false);

draw(scale3(4)*extrude("$\displaystyle\int\limits_{-\infty}^{+\infty}\!\! e^{-\alpha x^2}\!\!=\sqrt{\frac{\pi}{\alpha}}$",2Z),
     material(blue));

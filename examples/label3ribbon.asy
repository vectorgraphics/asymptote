import surface;

currentprojection=perspective(100,100,200,up=Y);

path[] g=texpath("$\displaystyle\int_{-\infty}^{+\infty}e^{-\alpha x^2}\,dx=
\sqrt{\frac{\pi}{\alpha}}$");

frame scene;
for(path tg:g) {
  surface[] surf=extrude(tg,2Z);
  for(surface s:surf)
    draw(scene,s,blue);
}

add3(scene,"test",10cm);
label(cameralink("test"),(50,-200));

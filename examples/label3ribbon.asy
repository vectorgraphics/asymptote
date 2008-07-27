import three;

settings.outformat="pdf";
currentprojection=perspective(100,100,200,up=Y);

path[] g=texpath("$\displaystyle\int_{-\infty}^{+\infty}e^{-\alpha x^2}\,dx=
\sqrt{\frac{\pi}{\alpha}}$");

for(path p:g)
  draw(extrude(p,2Z),blue);

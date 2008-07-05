import surface;

settings.outformat="pdf";
currentprojection=perspective(100,100,100,up=Y);
frame f;

label3(f,"$\displaystyle\int_{-\infty}^{+\infty} e^{-\alpha x^2}\,dx=
\sqrt{\frac{\pi}{\alpha}}$",blue);
add3(f,"test",10cm);
label(cameralink("test"),(50,-200));

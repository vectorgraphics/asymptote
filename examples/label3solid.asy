import three;

currentprojection=perspective(100,100,200,up=Y);

draw(scale3(4)*extrude("$\displaystyle\int_{-\infty}^{+\infty}
e^{-\alpha x^2}\,dx=\sqrt{\frac{\pi}{\alpha}}$",2Z),blue);

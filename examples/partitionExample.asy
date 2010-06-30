size(15cm);
import bezulate;

path[] p = texpath("$\sigma \Theta$");
pair m = min(p);
pair M = max(p);
real midy = 0.5(M.y+m.y);

path[] alpha = p[0:2];
path[] theta = p[2:5];
filldraw(p,lightgrey,black);

draw("{\tt partition}",(M.x+1mm,midy)--(M.x+5mm,midy),Arrow);
draw((M.x+1mm,midy+1mm)--(M.x+5mm,midy+2mm),Arrow);
draw((M.x+1mm,midy-1mm)--(M.x+5mm,midy-2mm),Arrow);

filldraw(shift((M.x+8.5mm,midy+3.5mm))*alpha,lightgrey,black);
filldraw(shift((M.x+5.5mm,0))*theta[0:2],lightgrey,black);
filldraw(shift(M.x+5.5mm,midy-2.5mm)*theta[2:3],lightgrey,black);

draw("{\tt merge}, {\tt bezulate}",(M.x+9mm,midy+3mm)--(M.x+15mm,midy+3mm),Arrow);
draw("{\tt merge}, {\tt bezulate}",(M.x+9mm,midy)--(M.x+15mm,midy),Arrow);
draw("{\tt bezulate}",(M.x+9mm,midy-2.5mm)--(M.x+15mm,midy-2.5mm),Arrow);

filldraw(shift(M.x+16mm-min(alpha).x,midy+3.5mm)*bezulate(alpha),lightgrey,black);
filldraw(shift(M.x+16mm-min(theta[0:2]).x,0)*bezulate(theta[0:2]),lightgrey,black);
filldraw(shift(M.x+16mm-min(theta[0:2]).x,midy-2.5mm)*bezulate(theta[2:3]),lightgrey,black);

// Original name : conicurv.mp 
// Author : L. Nobre G.
// Translators : J. Pienaar (2004) and John Bowman (2005)

import three;
texpreamble("\usepackage{bm}");

size(300,0);

currentprojection=perspective(10,-5,5.44); 

real theta=30, width=3, shortradius=2, bord=2, refsize=1, vecsize=2;
real height=0.3, anglar=1.75, totup=3;
real longradius=shortradius+width*Cos(theta), updiff=width*Sin(theta);

triple iplow=(0,shortradius,0), iphig=(0,longradius,updiff);
triple oplow=(-shortradius,0,0), ophig=(-longradius,0,updiff);
triple aplow=-iplow, aphig=(0,-longradius,updiff);
triple eplow=-oplow, ephig=(longradius,0,updiff);
triple anglebase=(0,longradius,0), centre=interp(iplow,iphig,0.5)+(0,0,height);
triple central=(0,0,centre.z), refo=(0,0.5*centre.y,centre.z);
triple refx=refsize*(0,Cos(theta),Sin(theta));
triple refy=refsize*(0,-Sin(theta),Cos(theta));

draw("$\theta$",arc(iplow,iplow+0.58*(iphig-iplow),anglebase),E);

draw(central,linewidth(2bp));
draw(iplow--iphig);
draw(oplow--ophig);
draw(aplow--aphig);
draw(eplow--ephig);
draw(iphig--anglebase--aplow,dashed);
draw(oplow--eplow,dashed);
draw(central--centre,dashed);

draw((0,0,-bord)--(0,longradius+bord,-bord)--(0,longradius+bord,totup)
     --(0,0,totup)--cycle);
draw(Label("$y$",1),refo--refo+refy,SW,Arrow3);
draw(Label("$x$",1),refo--refo+refx,SE,Arrow3);

draw(Label("$\vec{R}_N$",1),centre--centre+vecsize*refy,E,Arrow3);
draw(Label("$\vec{F}_a$",1),centre--centre+vecsize*refx,N,Arrow3);
draw(Label("$\vec{F}_c$",1),centre--centre+vecsize*Y,NE,Arrow3);
draw(Label("$\vec{P}$",1),centre--centre-vecsize*Z,E,Arrow3);
draw(Label("$\vec{v}$",1),centre--centre+vecsize*X,E,Arrow3);
draw(centre,10pt+blue);

draw(circle((0,0,updiff),longradius),linewidth(2bp));
draw(circle(O,shortradius),linewidth(2bp));

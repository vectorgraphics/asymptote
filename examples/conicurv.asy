// Original name : conicurv.mp 
// Author : L. Nobre G.
// Translator : J. Pienaar (2004)
// Y2K
import math;
import featpost3D;

texpreamble("
\usepackage{beton}
\usepackage{concmath}
\usepackage{ccfonts}
");

f = vector(10,-5,5.44);
Spread = 30;

real shortradius, longradius, theta, width, height;
real updiff, refsize, vecsize, anglar, bord, totup;
vector rn, fa, fc, pg, centre, speed, refx, refy;
vector iplow, iphig, oplow,ophig, anglebase, central, refo;
vector eplow, ephig, aplow,aphig;

theta = 30;
width = 3;
shortradius = 2;
bord = 2;
refsize = 1;
vecsize = 2;
height = 0.3;
anglar = 1.75;
totup = 3;

longradius = shortradius + width*Cos(theta);
updiff = width*Sin(theta);

iplow = vector(0,shortradius,0);
iphig = vector(0,longradius,updiff);
oplow = vector(-shortradius,0,0);
ophig = vector(-longradius,0,updiff);
aplow = -iplow;
aphig = vector(0,-longradius,updiff);
eplow = -oplow;
ephig = vector(longradius,0,updiff);

anglebase = vector(0,longradius,0);
centre = interp(iplow,iphig,0.5)+vector(0,0,height);
central = vector(0,0,Z(centre));

//	refo = (0,0,-shortradius*Sin(theta)/Cos(theta));
refo = vector(0,0.5*Y(centre),Z(centre));
refx = refsize*vector(0,Cos(theta),Sin(theta));
refy = refsize*vector(0,-Sin(theta),Cos(theta));

//	anglinen( iplow, oplow, (0,0,0), shortradius, "", 0 );
//	anglinen( iphig, ophig, (0,0,updiff), longradius, "", 0 );
angline(iphig,anglebase,iplow,width/anglar,"$\theta$",E);
draw(rp(central), linewidth(2));
draw(rp(iplow)--rp(iphig));
draw( rp(oplow)--rp(ophig));
draw( rp(aplow)--rp(aphig));
draw( rp(eplow)--rp(ephig));
draw( rp(iphig)--rp(anglebase)--rp(aplow), dashed);
draw( rp(oplow)--rp(eplow), dashed);
draw( rp(central)--rp(centre), dashed);
//	draw rp((0,-bord,-bord))--rp((0,longradius+bord,-bord))
//	   --rp((0,longradius+bord,totup))--rp((0,-bord,totup))--cycle
//	     withpen pencircle scaled 2pt;
draw( rp(vector(0,0,-bord))--rp(vector(0,longradius+bord,-bord))
    --rp(vector(0,longradius+bord,totup))--rp(vector(0,0,totup))--cycle);
draw (rp(refo)..rp(refo+refy), Arrow);
draw (rp(refo)..rp(refo+refx), Arrow);
label("$y$", rp(refo+refy), SW);
label("$x$", rp(refo+refx), SE);
rn = centre+vecsize*refy;
fa = centre+vecsize*refx;
fc = centre+vecsize*vector(0,1,0);
pg = centre+vecsize*vector(0,0,-1);
speed = centre+vecsize*vector(1,0,0);
draw( rp(centre)..rp(rn), Arrow);
draw( rp(centre)..rp(fa), Arrow);
draw( rp(centre)..rp(fc), Arrow);
draw( rp(centre)..rp(pg), Arrow);
draw( rp(centre)..rp(speed), Arrow);
label("$\vec{R}_N$", rp(rn), E);
label("$\vec{F}_a$", rp(fa), N);
label("$\vec{F}_c$", rp(fc), NE);
label("$\vec{P}$", rp(pg), E);
label("$\vec{v}$", rp(speed), E);
draw(rp(centre), linewidth(10pt)+blue);
currentpen = linewidth(2pt);
draw( rigorouscircle( vector(0,0,updiff), 
                      vector(0,0,1), 
                      longradius ) );
draw( rigorouscircle( vector(0,0,0), 
                      vector(0,0,1), 
                      shortradius ) );

shipout();


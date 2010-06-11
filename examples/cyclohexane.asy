import three;

currentprojection=perspective(300,-650,500);
currentlight.background=palecyan;

surface carbon=scale3(70)*unitsphere; // 70 pm
surface hydrogen=scale3(25)*unitsphere; // 25 pm

real alpha=90+aSin(1/3);

real CCbond=156; // 156 pm
real CHbond=110; // 110 pm

triple c1=(0,0,0);
triple h1=c1+CHbond*Z;
triple c2=rotate(alpha,c1,c1+Y)*(CCbond*Z);
triple h2=rotate(120,c1,c2)*h1;
triple h3=c2-CHbond*Z;
triple h4=rotate(120,c2,c1)*h3;

triple c3=rotate(120,c2,h3)*c1;
triple h5=c3+CHbond*Z;
triple h6=rotate(-120,c3,c2)*h5;

triple c4=rotate(-120,c3,h5)*c2;
triple h7=c4-CHbond*Z;
triple h8=rotate(120,c4,c3)*h7;

triple c5=rotate(120,c4,h7)*c3;
triple h9=c5+CHbond*Z;
triple h10=rotate(-120,c5,c4)*h9;

triple c6=rotate(-120,c5,h9)*c4;
triple h11=c6-CHbond*Z;
triple h12=rotate(120,c6,c5)*h11;

pen Black=gray(0.4);

defaultrender=render(compression=Zero,merge=true);

draw(shift(c1)*carbon,Black);
draw(shift(c2)*carbon,Black);
draw(shift(c3)*carbon,Black);
draw(shift(c4)*carbon,Black);
draw(shift(c5)*carbon,Black);
draw(shift(c6)*carbon,Black);


material White=material(diffusepen=gray(0.4),emissivepen=gray(0.6));

draw(shift(h1)*hydrogen,White);
draw(shift(h2)*hydrogen,White);
draw(shift(h3)*hydrogen,White);
draw(shift(h4)*hydrogen,White);
draw(shift(h5)*hydrogen,White);
draw(shift(h6)*hydrogen,White);
draw(shift(h7)*hydrogen,White);
draw(shift(h8)*hydrogen,White);
draw(shift(h9)*hydrogen,White);
draw(shift(h10)*hydrogen,White);
draw(shift(h11)*hydrogen,White);
draw(shift(h12)*hydrogen,White);


pen thick=linewidth(10);

draw(c1--c2--c3--c4--c5--c6--cycle,thick);

draw(c1--h1,thick);
draw(c1--h2,thick);
draw(c2--h3,thick);
draw(c2--h4,thick);
draw(c3--h5,thick);
draw(c3--h6,thick);
draw(c4--h7,thick);
draw(c4--h8,thick);
draw(c5--h9,thick);
draw(c5--h10,thick);
draw(c6--h11,thick);
draw(c6--h12,thick);

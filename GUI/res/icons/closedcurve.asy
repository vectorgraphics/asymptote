real pixel=1inch/96;
size(32*pixel,IgnoreAspect);
defaultpen(1.75bp);

path p=W..NW..ENE..0.5*SE..cycle;
draw(p);
dot(p,linewidth(8pixel));

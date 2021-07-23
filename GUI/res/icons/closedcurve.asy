defaultpen(2.5);

path p=W..NW..ENE..0.5*SE..cycle;
draw(p);
dot(p,linewidth(12));

shipout(pad(64,64));

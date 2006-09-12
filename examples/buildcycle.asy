size(200);

real w=1.35;

path[] p;
for(int k=0; k < 2; ++k) {
  int i=2+2*k;
  int ii=i^2;
  p[k]=(w/ii,1){1,-ii}::(w/i,1/i)::(w,1/ii){ii,-1};
}

path q0=(0,0)--(w,0.5);
path q1=(0,0)--(w,1.5);
draw(q0); draw(p[0]); draw(q1); draw(p[1]);
path s=buildcycle(q0,p[0],q1,p[1]);
fill(s,mediumgrey);

label("$P$",intersectionpoint(p[0],q0),N);
label("$Q$",intersectionpoint(p[0],q1),E);
label("$R$",intersectionpoint(p[1],q1),W);
label("$S$",intersectionpoint(p[1],q0),S);
label("$f > 0$",0.5*(min(s)+max(s)),UnFill);

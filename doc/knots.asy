import syzygy;

Braid initial;
initial.n = 4;
initial.add(bp,1);
initial.add(bp,0);
initial.add(bp,2);
initial.add(bp,1);
initial.add(phi,2);
initial.add(phi,0);

Syzygy pp;
pp.lsym="\Phi\Phi"; pp.codename="PhiAroundPhi";

pp.number=true;
pp.initial=initial;
pp.apply(r4b,2,1);
pp.apply(r4b,0,0);
pp.apply(r4a,1,0);
pp.swap(0,1);
pp.apply(-r4b,1,0);
pp.apply(-r4a,0,1);
pp.apply(-r4a,2,0);
pp.swap(4,5);

pp.draw();


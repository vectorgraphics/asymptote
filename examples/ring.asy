size(0,100);
path unitcircle=E..N..W..S..cycle;
path g=scale(2)*unitcircle;
fill(unitcircle^^g,yellow+evenodd,(0,0),1.0,yellow+brown,(0,0),2);

shipout();

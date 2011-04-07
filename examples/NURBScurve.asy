import three;

size(10cm);

currentprojection=perspective(50,80,50);

// Nonrational curve:
// udegree=3, nu=6;
real[] knot={0,0,0,0,0.4,0.6,1,1,1,1};

triple[] P={
  (-31.2061,12.001,6.45082),
  (-31.3952,14.7353,6.53707),
  (-31.5909,21.277,6.70051),
  (-31.4284,25.4933,6.76745),
  (-31.5413,30.3485,6.68777),
  (-31.4896,32.2839,6.58385)
  };

draw(P,knot,green);

// Rational Bezier curve:
// udegree=3, nu=4;
real[] knot={0,0,0,0,1,1,1,1};
path3 g=scale3(20)*(X{Y}..{-X}Y);
triple[] P={point(g,0),postcontrol(g,0),precontrol(g,1),point(g,1)};

// Optional weights:
real[] weights=array(P.length,1.0);
weights[2]=5;

draw(P,knot,weights,red);


import three;

size(10cm);
currentlight=Viewport;

surface s=surface(patch(new triple[][] {
      {(0,0,0),(1,0,0),(1,0,0),(2,0,0)},
      {(0,1,0),(1,0,1),(1,0,1),(2,1,0)},
      {(0,1,0),(1,0,-1),(1,0,-1),(2,1,0)},
      {(0,2,0),(1,2,0),(1,2,0),(2,2,0)}}));

draw(s,yellow);
draw(s.s[0].vequals(0.5),squarecap+2bp+blue,currentlight);
draw(s.s[0].uequals(0.5),squarecap+2bp+red,currentlight);

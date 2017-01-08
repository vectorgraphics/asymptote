import three;
currentlight=Viewport;

size(10cm);

surface s=surface(patch(new triple[][] {
      {(0,0,0),(1,0,0),(1,0,0),(2,0,0)},
      {(0,1,0),(1,0,1),(1,0,1),(2,1,0)},
      {(0,1,0),(1,0,-1),(1,0,-1),(2,1,0)},
      {(0,2,0),(1,2,0),(1,2,0),(2,2,0)}}));

s.s[0].colors=new pen[] {red,green,blue,black};
draw(s,nolight);

surface t=shift(Z)*unitplane;
t.s[0].colors=new pen[] {red,green,blue,black};

draw(t,nolight);

size(15cm,0);

pair d=(1.5,1);
real s=d.x+1;

picture box(string s) {
  picture pic=new picture;
  draw(pic,box(0,d));
  label(pic,s,d/2);
  return pic;
}

add(box("$k_1$"));
add(shift(s)*box("$k_2$"));
add(shift(s)^2*box("$k_3$"));

guide g=(d.x,d.y/2)--(s,d.y/2);
guide G=(d.x/2,-(s-d.x))--(d.x/2,0);

draw(baseline("$\ldots$"),shift(-s)*g,BeginArrow,BeginPenMargin);
draw("$Z_1$",g,BeginArrow,BeginPenMargin);
draw("$E_1$",g,LeftSide,Blank);
draw("$Z_3$",shift(s)*g,Arrow,PenMargin);
draw("$E_3$",shift(s)*g,LeftSide,Blank);
draw("$Z_2$",shift(s)*G,Arrow,PenMargin);
draw("$E_2$",shift(s)*G,LeftSide,Blank);
draw(baseline("$\ldots$"),shift(s)^2*g,Arrow,PenMargin);


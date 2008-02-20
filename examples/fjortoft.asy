size(15cm,0);

pair d=(1.5,1);
real s=d.x+1;

picture box(string s) {
  picture pic;
  draw(pic,box(0,d));
  label(pic,s,d/2);
  return pic;
}

add(box("$k_1$"));
add(shift(s)*box("$k_2$"));
add(shift(s)^2*box("$k_3$"));

path g=(d.x,d.y/2)--(s,d.y/2);
path G=(d.x/2,-(s-d.x))--(d.x/2,0);

draw(Label(baseline("$\ldots$")),shift(-s)*g,BeginArrow,BeginPenMargin);
draw(Label("$Z_1$"),g,BeginArrow,BeginPenMargin);
draw(Label("$E_1$",LeftSide),g,Blank);
draw(Label("$Z_3$"),shift(s)*g,Arrow,PenMargin);
draw(Label("$E_3$",LeftSide),shift(s)*g,Blank);
draw(Label("$Z_2$"),shift(s)*G,Arrow,PenMargin);
draw(Label("$E_2$",LeftSide),shift(s)*G,Blank);
draw(Label(baseline("$\ldots$")),shift(s)^2*g,Arrow,PenMargin);

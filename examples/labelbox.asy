real margin=2mm;
frame b1,b2;
box(b1,Label("small box",(0,0)),margin);
box(b2,Label("LARGER BOX",(0,-2cm)),margin);
add(b1);
add(b2);
draw(point(b1,S)--point(b2,N),currentpen);

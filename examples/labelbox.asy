real margin=2mm;
frame b1=labelBox(margin,"small box",(0,0));
frame b2=labelBox(margin,"LARGER BOX",(0,-2cm));
frame f;
add(f,b1);
add(f,b2);
draw(f,point(b1,S)--point(b2,N),currentpen);

addabout((0,0),f);

shipout();

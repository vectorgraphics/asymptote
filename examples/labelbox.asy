real margin=2mm;
frame b1,b2;
labelbox(b1,margin,"small box",(0,0));
labelbox(b2,margin,"LARGER BOX",(0,-2cm));
add(b1);
add(b2);
draw(point(b1,S)--point(b2,N),currentpen);

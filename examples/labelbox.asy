real margin=2mm;
frame b1=labelBox(margin,"small box",(0,0));
frame b2=labelBox(margin,"LARGER BOX",(0,-2cm));
add(b1);
add(b2);
draw(point(b1,S)--point(b2,N),currentpen);

size(200);

pen indigo=rgb(102/255,0,238/255);

void rainbow(path g) {
  draw(new path[] {scale(1.3)*g,scale(1.2)*g,scale(1.1)*g,g,
	scale(0.9)*g,scale(0.8)*g,scale(0.7)*g},
    new pen[] {red,orange,yellow,green,blue,indigo,purple});
}

rainbow((1,0){N}..(0,1){W}..{S}(-1,0));
rainbow(scale(4)*shift(-0.5,-0.5)*unitsquare);

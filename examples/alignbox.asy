real margin=1mm;

void prodShapeIn(picture pic=currentpicture,
                 string leftString, string rightString) {
  object leftObject=align(object(leftString,ellipse,1.5*margin),W);
  add(pic,leftObject);
  object rightObject=align(object(rightString,ellipse,1.5*margin),4E);
  add(pic,rightObject);
  pic.add(new void(frame f, transform t) {
      draw(f,point(leftObject,NE,t)--point(rightObject,W,t));
    });
}

prodShapeIn("$x^2$", "$\sin x$");

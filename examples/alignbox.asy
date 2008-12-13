real margin=1.5mm;

object leftObject=align(object("$x^2$",ellipse,margin),W);
add(leftObject);
object rightObject=align(object("$\sin x$",ellipse,margin),4E);
add(rightObject);
currentpicture.add(new void(frame f, transform t) {
    draw(f,point(leftObject,NE,t)--point(rightObject,W,t));
  });

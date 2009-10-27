real margin=1.5mm;

object left=align(object("$x^2$",ellipse,margin),W);
add(left);
object right=align(object("$\sin x$",ellipse,margin),4E);
add(right);
add(new void(frame f, transform t) {
    draw(f,point(left,NE,t)--point(right,W,t));
  });

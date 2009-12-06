size(0,100);
real margin=2mm;
pair z1=(0,1);
pair z0=(0,0);

object Box=draw("small box",box,z1,margin);
object Ellipse=draw("LARGE ELLIPSE",ellipse,z0,margin);

add(new void(frame f, transform t) {
    draw(f,point(Box,SW,t){SW}..{SW}point(Ellipse,NNE,t));
  });

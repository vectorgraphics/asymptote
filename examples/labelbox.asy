size(0,100);
real margin=2mm;
pair z1=(0,1);
pair z0=(0,0);

object label1=draw("small box",box,z1,margin);
object label0=draw("LARGE ELLIPSE",ellipse,z0,margin);

add(new void(frame f, transform t) {
    draw(f,point(label1,SW,t){SW}..{SW}point(label0,NNE,t));
  });

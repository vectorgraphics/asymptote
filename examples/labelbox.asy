size(0,100);
real margin=2mm;
pair z1=(0,1);
pair z0=(0,0);

object label1=object(box,Label("small box",z1),margin);
object label0=object(ellipse,Label("LARGE ELLIPSE",z0),margin);

add(new void(frame f, transform t) {
    draw(f,point(label1,S,t)--point(label0,N,t));
  });

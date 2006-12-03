size(0,100);
real margin=2mm;
pair z1=(0,1);
pair z0=(0,0);

labelframe label1=labelframe(box,Label("small box",z1),margin);
labelframe label0=labelframe(ellipse,Label("LARGE ELLIPSE",z0),margin);

currentpicture.add(new void(frame f, transform t) {
    draw(f,point(label1,S,t)--point(label0,N,t));
  });

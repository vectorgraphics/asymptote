import pdfanim;

animation a;

int n=20;
for(int i=0; i < 3.5n; ++i) {
  picture pic;
  size(pic,100);
  draw(pic,circle((0,sin(pi/n*i)),1));
  a.add(pic);
}

label(a.pdf());
label(a.controlpanel(),truepoint(S),S);
pair z=truepoint(S);
label(a.progress(blue),z,SW);
label(a.delay(red),z,SE);

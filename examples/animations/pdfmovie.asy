import animate;
import patterns;

settings.tex="pdflatex";

animation a;

add("brick",brick(black));

int n=20;
for(int i=0; i < 3.5n; ++i) {
  picture pic;
  size(pic,100);
  path g=circle((0,sin(pi/n*i)),1);
  fill(pic,g,mediumred);
  fill(pic,g,pattern("brick"));
  a.add(pic);
}

label(a.pdf("controls",multipage=false));

size(250,250);
import graph;
import palette;

file fin=single(xinput("image.xdr"));
real[][][] v=read3(fin);

pen[] Palette=BWRainbow();

image(v[0],Palette,(0,0),(1,-1));
addabout((1,-1),palette(v[0],Palette,"$Q$",
	LeftTicks(0.01,0.0,Ticksize,0.0,"%+#.2f")));


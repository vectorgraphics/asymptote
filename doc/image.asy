size(200,200);
import palette;
import graph;

file fin=single(xinput("weiss"));
real[][][] v=read3(fin);

pen[] Palette=BWrainbow(true);

image(v[0],Palette,(0,0),(1,-1));
addabout((1,-1),
	 palette(v[0],Palette,LeftTicks(0.01,0.0,Ticksize,0.0,"%+#.2f")));

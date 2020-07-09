size(200);
import solids;

currentprojection=perspective(4,4,3);
revolution hyperboloid=revolution(graph(new triple(real z) {
      return (sqrt(1+z*z),0,z);},-2,2,20,operator ..),axis=Z);
draw(surface(hyperboloid),green,render(compression=Low,merge=true));
draw(hyperboloid,6,blue+0.15mm,longitudinalpen=nullpen);

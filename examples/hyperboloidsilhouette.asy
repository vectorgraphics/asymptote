size(200);
import solids;
settings.render=0;
settings.prc=false;

currentprojection=perspective(4,4,3);
revolution hyperboloid=revolution(graph(new triple(real z) {
      return (sqrt(1+z*z),0,z);},-2,2,20,operator ..),axis=Z);
draw(hyperboloid.silhouette(64),blue);

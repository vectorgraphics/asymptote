size(200);

pen[][] p={{red,green,blue,cyan},{blue,green,magenta,rgb(black)}};
path G=(0,0){dir(-120)}..(1,0)..(1,1)..(0,1)..cycle;
path[] g={G,subpath(G,2,1)..(2,0)..(2,1)..cycle};
pair[][] z={{(0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5)},{(2,0.5),(2,0.5),(1.5,0.5),(2,0.5)}};
tensorshade(g,p,z);

dot(g);

size(200);

pen[][] p={{red,green,blue,cyan},{green,blue,rgb(black),magenta}};
path G=(0,0){dir(-120)}..(1,0)..(1,1)..(0,1)..cycle;
path[] g={G,subpath(G,1,2)..(2,1)..(2,0)..cycle};
pair[][] z={{(0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5)},{(2,0.5),(2,0.5),(1.5,0.5),(2,0.5)}};
tensorshade(g,p,z);
dot(g);

size(200);

pen[][] p={{red,green,blue,magenta}};
path g=(0,0){dir(45)}..(1,0)..(1,1)..(0,1)..cycle;
pair[][] z={{(0.5,0.5),(0.5,0.5),(0.5,0.5),(0.5,0.5)}};
tensorshade(g,p,g,z);
dot(g);

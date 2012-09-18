import three;
size(10cm);

triple[] v={O,X,X+Y,Y};

triple[] n={Z,X};

int[][] vi={{0,1,2},{2,3,0}};
int[][] ni={{0,0,0},{1,1,1}};

pen[] p={red+opacity(0.5),green+opacity(0.5),blue+opacity(0.5),
         black+opacity(0.5)};

// Adobe Reader exhibits a PRC rendering bug for opacities in (0.5,1):
//pen[] p={red+opacity(0.9),green+opacity(0.9),blue+opacity(0.9),black+opacity(0.9)};

int[][] pi={{0,1,2},{2,3,0}};
draw(v,vi,n,ni,red);
draw(v+Z,vi,p,pi);

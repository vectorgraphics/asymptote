import three;
size(10cm);

currentlight=Headlamp;

triple[] v={O,X,X+Y,Y};

triple[] n={Z,X};

int[][] vi={{0,1,2},{2,3,0}};
int[][] ni={{1,0,1},{1,1,1}};

// Adobe Reader exhibits a PRC rendering bug for opacities:
pen[] p={red+opacity(0.5),green+opacity(0.5),blue+opacity(0.5),
         black+opacity(0.5)};

int[][] pi={{0,1,2},{2,3,0}};
draw(v,vi,n,ni,red);
draw(v+Z,vi,n,ni,p,pi);
//draw(v+Z,vi,p,pi);
//draw(v,vi,red);
//draw(v+Z,vi);

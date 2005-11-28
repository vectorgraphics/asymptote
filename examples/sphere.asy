import light;

size(150);

pen color(triple v) {
  return currentlight.intensity(v)*red;
}

real s=360/64;

int[] edges={0,0,0,2};

for(real i=0; i < 180; i += s) {
  for(real j=0; j < 360; j += s) {
    if(dot(dir(i+s/2,j+s/2),currentprojection.camera) >= 0) {
      triple[] v={dir(i,j),dir(i,j+s),dir(i+s,j+s),dir(i+s,j)};
      pen[] p={color(v[0]),color(v[1]),color(v[2]),color(v[3])};
      gouraudshade(v[0]--v[1]--v[2]--v[3]--cycle3,p,v,edges);
    }
  }
}

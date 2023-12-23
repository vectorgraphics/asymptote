import three;

size(300);

patch p=patch(unstraighten(unitplane.s[0].external()));

p.P[3][0]+=(0,0,1);

p.P[1][0]+=(0,0,1/3);
p.P[2][0]+=(0,0,2/3);
p.P[3][1]+=(0,0,2/3);
p.P[3][2]+=(0,0,1/3);

p.P[2][1]=interp(p.P[2][0],p.P[2][3],1/3);
p.P[2][2]=interp(p.P[2][0],p.P[2][3],2/3);

p.P[1][1]=interp(p.P[1][0],p.P[1][3],1/3);
p.P[1][2]=interp(p.P[1][0],p.P[1][3],2/3);

draw(surface(p),red+opacity(0.75));

void dot(triple[][] P) {
  for(int i=0; i < 4; ++i)
    for(int j=0; j < 4; ++j) {
      draw(string(i)+","+string(j),P[i][j],linewidth(1mm));
    }
}

dot(surface(p).s[0].P);

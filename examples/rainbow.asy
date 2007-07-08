size(200);

void draw(path[] g, pen[] p) {
  path[] G;
  pen[][] P;
  string differentlengths="arrays have different lengths";
  if(g.length != p.length) abort(differentlengths);
  for(int i=0; i < g.length-1; ++i) {
    path g0=g[i];
    path g1=g[i+1];
    if(length(g0) != length(g1)) abort(differentlengths);
    for(int j=0; j < length(g0); ++j) {
      G.push(subpath(g0,j,j+1)--reverse(subpath(g1,j,j+1))--cycle);
      P.push(new pen[] {p[i],p[i],p[i+1],p[i+1]});
    }
  }
  tensorshade(G,P);
}

pen indigo=rgb(102/255,0,238/255);

void rainbow(path g) {
  draw(new path[] {scale(1.3)*g,scale(1.2)*g,scale(1.1)*g,g,
	scale(0.9)*g,scale(0.8)*g,scale(0.7)*g},
    new pen[] {red,orange,yellow,green,blue,indigo,purple});
}

rainbow((1,0){N}..(0,1){W}..{S}(-1,0));
rainbow(scale(4)*shift(-0.5,-0.5)*unitsquare);

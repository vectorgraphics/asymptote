size(200);
int np=100;
pair[] points;

real r() {return 1.2*(rand()/randMax*2-1);}

for(int i=0; i < np; ++i)
  points.push((r(),r()));

int[][] trn=triangulate(points);

for(int i=0; i < trn.length; ++i) {
  draw(points[trn[i][0]]--points[trn[i][1]]);
  draw(points[trn[i][1]]--points[trn[i][2]]);
  draw(points[trn[i][2]]--points[trn[i][0]]);
}

for(int i=0; i < np; ++i)
  dot(points[i],red);

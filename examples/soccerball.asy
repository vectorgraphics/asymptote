import graph3; 
size(400); 
currentlight.background=palegreen;

defaultrender=render(compression=Zero,merge=true);

real c=(1+sqrt(5))/2; 
 
triple[] z={(c,1,0),(-c,1,0),(-c,-1,0),(c,-1,0)}; 
triple[] x={(0,c,1),(0,-c,1),(0,-c,-1),(0,c,-1)}; 
triple[] y={(1,0,c),(1,0,-c),(-1,0,-c),(-1,0,c)}; 
 
triple[][] Q={
  {z[0],y[1],x[3],x[0],y[0],z[3]},
  {z[1],x[0],x[3],y[2],z[2],y[3]},
  {z[2],z[1],y[2],x[2],x[1],y[3]},
  {z[3],z[0],y[0],x[1],x[2],y[1]},
  {x[0],x[3],z[1],y[3],y[0],z[0]},
  {x[1],x[2],z[2],y[3],y[0],z[3]},
  {x[2],x[1],z[3],y[1],y[2],z[2]},
  {x[3],x[0],z[0],y[1],y[2],z[1]},
  {y[0],y[3],x[1],z[3],z[0],x[0]},
  {y[1],y[2],x[2],z[3],z[0],x[3]},
  {y[2],y[1],x[3],z[1],z[2],x[2]},
  {y[3],y[0],x[0],z[1],z[2],x[1]} 
}; 
 
path3 p=arc(O,Q[0][0],Q[0][1]); 
real R=abs(point(p,reltime(p,1/3))); 
 
triple[][] P;
for(int i=0; i < Q.length; ++i){
  P[i]=new triple[] ; 
  for(int j=0; j < Q[i].length; ++j){
    P[i][j]=Q[i][j]/R; 
  } 
} 
 
surface sphericaltriangle(triple center, triple A, triple B, triple C,
                          int nu=3, int nv=nu) {
  path3 tri1=arc(center,A,B); 
  path3 tri2=arc(center,A,C); 
  path3 tri3=arc(center,B,C); 
  triple tri(pair p) {
    path3 cr=arc(O,relpoint(tri2,p.x),relpoint(tri3,p.x)); 
    return relpoint(cr,p.y); 
  }; 
 
  return surface(tri,(0,0),(1-sqrtEpsilon,1),nu,nv,Spline); 
} 
 
for(int i=0; i < P.length; ++i){
  triple[] pentagon=sequence(new triple(int k) {
      path3 p=arc(O,P[i][0],P[i][k+1]); 
      return point(p,reltime(p,1/3)); 
    },5); 
  pentagon.cyclic=true; 
  draw(sequence(new path3(int k) {
        return arc(O,pentagon[k],pentagon[k+1]);},5),linewidth(2pt)); 
  triple M=unit(sum(pentagon)/5); 
  for(int i=0; i < 5; ++i){
    surface sf=sphericaltriangle(O,pentagon[i],M,pentagon[i+1]); 
    draw(sf,black); 
  } 
} 
 
for(int i=0; i < P.length; ++i){
  for(int j=1; j <= 5; ++j){
    triple K=P[i][0]; 
    triple A=P[i][j]; 
    triple B=P[i][(j % 5)+1]; 
    path3[] p={arc(O,K,A),arc(O,A,B),arc(O,B,K)}; 
    draw(subpath(p[0],reltime(p[0],1/3),reltime(p[0],2/3)),linewidth(4pt)); 
    triple[] hexagon={point(p[0],reltime(p[0],1/3)),
                      point(p[0],reltime(p[0],2/3)),
                      point(p[1],reltime(p[1],1/3)),
                      point(p[1],reltime(p[1],2/3)),
                      point(p[2],reltime(p[2],1/3)),
                      point(p[2],reltime(p[2],2/3))}; 
    hexagon.cyclic=true; 
    triple M=unit(sum(hexagon)/6); 
    for(int i=0; i < 6; ++i){
      surface sf=sphericaltriangle(O,hexagon[i],M,hexagon[i+1]); 
      draw(sf,white); 
    } 
  } 
}

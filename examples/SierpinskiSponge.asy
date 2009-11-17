size(200);
import palette;
import three;
    
triple[] M={
  (-1,-1,-1),(0,-1,-1),(1,-1,-1),(1,0,-1),
  (1,1,-1),(0,1,-1),(-1,1,-1),(-1,0,-1),
  (-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0),
  (-1,-1,1),(0,-1,1),(1,-1,1),(1,0,1),
  (1,1,1),(0,1,1),(-1,1,1),(-1,0,1)
};
    
int level=3;
    
surface s;
    
void recur(triple p, real u, int l) {
  if(l < level)
    for(triple V : M)
      recur(p+u*V,u/3,l+1);
  else
    for(triple V : M) {
      s.append(surface((p+u*(V+.5(1,-1,-1)))--(p+u*(V+.5(1,1,-1)))
                       --(p+u*(V+.5(1,1,1)))--(p+u*(V+.5(1,-1,1)))--cycle));
      s.append(surface((p+u*(V+.5(1,1,-1)))--(p+u*(V+.5(-1,1,-1)))
                       --(p+u*(V+.5(-1,1,1)))--(p+u*(V+.5(1,1,1)))--cycle));
      s.append(surface((p+u*(V+.5(-1,1,-1)))--(p+u*(V+.5(-1,-1,-1)))
                       --(p+u*(V+.5(-1,-1,1)))--(p+u*(V+.5(-1,1,1)))--cycle));
      s.append(surface((p+u*(V+.5(-1,-1,-1)))--(p+u*(V+.5(1,-1,-1)))
                       --(p+u*(V+.5(1,-1,1)))--(p+u*(V+.5(-1,-1,1)))--cycle));
      s.append(surface((p+u*(V+.5(1,-1,1)))--(p+u*(V+.5(1,1,1)))
                       --(p+u*(V+.5(-1,1,1)))--(p+u*(V+.5(-1,-1,1)))--cycle));
      s.append(surface((p+u*(V+.5(1,-1,-1)))--(p+u*(V+.5(-1,-1,-1)))
                       --(p+u*(V+.5(-1,1,-1)))--(p+u*(V+.5(1,1,-1)))--cycle));
    }
}
    
recur((0,0,0),1/3,1);
    
s.colors(palette(s.map(abs),Rainbow()));
    
draw(s);

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
    
triple[] K={
  (1,-1,-1),(1,0,-1),(1,1,-1),(1,1,0),
  (1,1,1),(1,0,1),(1,-1,1),(1,-1,0),
  (0,0,-1),(0,0,1),(0,-1,0),(0,1,0)
};
    
surface s;
    
void recur(triple p, real u, int level) {
  if(level > 1 )
    for(triple V : M)
      recur(p+u*V,u/3,level-1);
  else
    for(triple V : K) {
      transform3 T=shift(p)*scale3(u)*shift(V)*scale3(0.5);
      surface t=T*surface((1,-1,-1)--(1,1,-1)--(1,1,1)--(1,-1,1)--cycle);
      s.append(t);
      s.append(scale3(-1)*t);
    }
}
    
recur((0,0,0),1/3,3);
    
surface sf;
sf.append(s);
sf.append(rotate(90,Y)*s);
sf.append(rotate(90,Z)*s);
sf.colors(palette(sf.map(abs),Rainbow()));
    
draw(sf,render(compression=Low,merge=true));

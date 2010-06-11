size(200);
import palette;
import three;
currentprojection=perspective(8,2,1);
    
triple[] M={(0,0,1),1/3*(sqrt(8),0,-1),
            1/3*((sqrt(8))*Cos(120),(sqrt(8))*Sin(120),-1),
            1/3*((sqrt(8))*Cos(240),(sqrt(8))*Sin(240),-1)};
    
int level=5;
    
surface s;
    
void recur(triple p, real u, int l) {
  if(l < level)
    for(triple V : M)
      recur(p+u*V,u/2,l+1);
  else
    for(triple V : M) {
      s.append(surface((p+u*(V+M[0]))--(p+u*(V+M[1]))--(p+u*(V+M[2]))--cycle));
      s.append(surface((p+u*(V+M[0]))--(p+u*(V+M[2]))--(p+u*(V+M[3]))--cycle));
      s.append(surface((p+u*(V+M[0]))--(p+u*(V+M[3]))--(p+u*(V+M[1]))--cycle));
      s.append(surface((p+u*(V+M[3]))--(p+u*(V+M[2]))--(p+u*(V+M[1]))--cycle));
    }
}
    
recur(O,0.5,1);
    
s.colors(palette(s.map(zpart),Rainbow()));
    
draw(s,render(merge=true));

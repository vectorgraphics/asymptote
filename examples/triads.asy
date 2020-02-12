import graph;

path p=(10,75)..(15,85)..(20,90)..(35,85)..(40,79)--(78,30)..(85,15)..(87,5);

pair l=point(p,3.5);
pair m=point(p,4.5);
pair s=point(p,4.9);

pen c=linewidth(1.5);
pair o=(m.x,0.5(m.x+l.y));

pen d=c+darkgreen;

void drawarrow(string s="", pair p, pair q, side side=RightSide,
	       bool upscale=false, pen c)
{
  path g=p{dir(-5)}..{dir(-85)}q;
  if(upscale) g=reverse(g); 
  draw(s,g,side,c,Arrow(Fill,0.65));
} 

void spectrum(pair l,pair m, pair s) {
  draw(p,c);
 
  d += 4.0;
  dot("$p$",l,SW,d); 
  dot("$q$",m,SW,d);
  dot("$k$",s,SW,d);

  xaxis("$k$",0);
  yaxis("$E(k)$",0);
}

drawarrow("$T_p$",l,m,true,blue);
drawarrow("$T_k$",m,s,LeftSide,red);
spectrum(l,m,s);
shipout("triadpqk");

erase();

drawarrow("$-T_p$",l,m,LeftSide,red);
drawarrow("$-T_q$",m,s,true,blue);
spectrum(l,s,m);
shipout("triadpkq");

erase();

drawarrow("$T_k$",l,m,true,blue);
drawarrow("$T_q$",m,s,LeftSide,red);
spectrum(m,s,l);

shipout("triadkpq");

erase();

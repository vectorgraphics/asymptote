import graph;

path p=(10,75)..(15,85)..(20,90)..(35,85)..(40,79)--(78,30)..(85,15)..(87,5);

pair l=point(p,3.5);
pair m=point(p,4.5);
pair s=point(p,4.9);

pen c=linewidth(1.5);
pair o=(xpart(m),0.5(ypart(m)+ypart(l)));

pen d=c+darkgreen;

void drawarrow(pair p, pair q, bool upscale=false, pen c)
{
  path g=p{dir(-5)}..{dir(-85)}q;
  if(upscale) g=reverse(g); 
  draw(g,c,Arrow(Fill,0.65));
} 

void spectrum(pair l,pair m, pair s) {
  draw(p,c);
 
  labeldot(4,"$p$",l,SW,d); 
  labeldot(4,"$q$",m,SW,d);
  labeldot(4,"$k$",s,SW,d);

  xaxis(0,"$k$");
  yaxis(0,"$E(k)$");
}

drawarrow(l,m,true,blue);
drawarrow(m,s,red);
spectrum(l,m,s);
shipout("triadpqk");

erase();

drawarrow(l,m,red);
drawarrow(m,s,true,blue);
spectrum(l,s,m);
shipout("triadpkq");

erase();

drawarrow(l,m,true,blue);
drawarrow(m,s,red);
spectrum(m,s,l);

shipout("triadkpq");

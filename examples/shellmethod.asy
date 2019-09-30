import graph3;
import solids;

size(400);
currentprojection=perspective(2,3,30,up=Y);
currentlight=light(gray(0.75),(0.25,-0.25,1),(0,1,0));

pen color=green;
real alpha=240;

real f(real x) {return 2x^2-x^3;}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}

int n=10;
path3[] blocks=new path3[n];
for(int i=1; i <= n; ++i) {
  real height=f((i-0.5)*2/n);
  real left=(i-1)*2/n;
  real right=i*2/n;
  blocks[i-1]=
    (left,0,0)--(left,height,0)--(right,height,0)--(right,0,0)--cycle;
}

path p=graph(F,0,2,n,operator ..)--cycle;
surface s=surface(p);
path3 p3=path3(p);

render render=render(compression=0,merge=true);

revolution a=revolution(p3,Y,0,alpha);
draw(surface(a),color,render);
draw(rotate(alpha,Y)*s,color,render);
for(int i=0; i < n; ++i)
  draw(surface(blocks[i]),color+opacity(0.5),black,render);
draw(p3);

xaxis3(Label("$x$",1,align=2X),Arrow3);
yaxis3(Label("$y$",1,align=2Y),ymax=1.4,dashed,Arrow3);
arrow("$y=2x^2-x^3$",XYplane(F(1.8)),X+Z,1.5cm,red,Arrow3(DefaultHead2));
draw(arc(1.17Y,0.3,90,0,7.5,180),ArcArrow3);

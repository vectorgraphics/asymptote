import graph;

size(9cm,8cm,IgnoreAspect);
string data="westnile.csv";

file in=line(csv(input(data)));

string[] columnlabel=in;

real[][] A=dimension(in,0,0);
A=transpose(A);
real[] number=A[0], survival=A[1];

guide g=graph(number,survival);
draw(g);

xaxis("Initial no.\ of mosquitoes per bird ($S_{M_0}/N_{B_0}$)",
      Bottom,LeftTicks);
xaxis(Top);
yaxis("Susceptible bird survival",Left,RightTicks);
yaxis(Right);

real a=number[0];
real b=number[-1];

real y1=0.9;
path h1=(a,y1)--(b,y1);
real n1=interp(a,b,intersect(h1,g).x);

real y2=0.48;
path h2=(a,y2)--(b,y2);
real n2=interp(a,b,intersect(h2,g).x);

draw((a,y1)--(n1,y1)--(n1,0),dotted+1bp);
draw((a,y2)--(n2,y2)--(n2,0),dashed);

pen p=fontsize(10);

real y3=0.043;
path reduction=(n2,y3)--(n1,y3);
draw(reduction,Arrow);

arrow(minipage("\flushleft{\begin{itemize}\item[1.] Estimate proportion of 
birds surviving at end of season\end{itemize}}",
	       100),(n2,y2),NNE,1cm,(-20,5),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[2.] Read off initial mosquito
abundance\end{itemize}}",80),(n2,0),NE,2.0cm,(-52,6),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[3.] Determine desired bird
survival for next season\end{itemize}}",
	       90),(n1,y1),SW,arrowlength,(16,-2),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[4.] Calculate required
proportional reduction in mosquitoes\end{itemize}}",
	       90),point(reduction,0.5),NW,1.4cm,(3,-27),p,NoFill);
shipout();

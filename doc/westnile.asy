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

real S1=0.475;
path h1=(a,S1)--(b,S1);
real M1=interp(a,b,intersect(h1,g).x);

real S2=0.9;
path h2=(a,S2)--(b,S2);
real M2=interp(a,b,intersect(h2,g).x);

labelx("$M_1$",M1);
labelx("$M_2$",M2);

draw((a,S2)--(M2,S2)--(M2,0),Dotted);
draw((a,S1)--(M1,S1)--(M1,0),dashed);

pen p=fontsize(10);

real y3=0.043;
path reduction=(M1,y3)--(M2,y3);
draw(reduction,Arrow,TrueMargin(0,0.5*(linewidth(Dotted)+linewidth())));

arrow(minipage("\flushleft{\begin{itemize}\item[1.] Estimate proportion of 
birds surviving at end of season\end{itemize}}",
	       100),(M1,S1),NNE,1cm,(-20,5),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[2.] Read off initial mosquito
abundance\end{itemize}}",80),(M1,0),NE,2.0cm,(-49,6),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[3.] Determine desired bird
survival for next season\end{itemize}}",
	       90),(M2,S2),SW,arrowlength,(16,-2),p,NoFill);

arrow(minipage("\flushleft{\begin{itemize}\item[4.] Calculate required
proportional reduction in mosquitoes\end{itemize}}",
	       90),point(reduction,0.5),NW,1.5cm,(3,-27),p,NoFill);
shipout();

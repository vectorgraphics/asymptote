size(0,200);
import geometry;

real A=130;
real B=40;

pair O=(0,0); 
pair R=(1,0);
pair P=dir(A);
pair Q=dir(B);

draw(circle(O,1.0));
draw(Q--O--P);
draw(P--Q,red);
draw(O--Q--R--cycle);

draw("$A$",arc(R,O,P,0.3),blue,Arrow,PenMargin);
draw("$B$",arc(R,O,Q,0.6),blue,Arrow,PenMargin);
pair S=(Cos(B),0);
draw(Q--S,blue);
perpendicular(S,NE,blue);

dot(O);
dot("$R=(1,0)$",R);
dot("$P=(\cos A,\sin A)$",P,dir(O--P)+W);
dot("$Q=(\cos B,\sin B)$",Q,dir(O--Q));

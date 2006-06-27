texpreamble("
\usepackage{bm}
\def\v{\bm}
\def\grad{\v\nabla}
\def\cross{{\v\times}}
\def\curl{\grad\cross}
\def\del{\nabla}
");

defaultpen(fontsize(10pt));

real margin=1.5mm;
real h=8cm;
real v=4cm;

Label cell(string s, string size="", pair position,
	    align align=NoAlign, pen p=nullpen, filltype filltype=NoFill)
{
  return Label(s,size,realmult(position,(h,v)),align,p,filltype);
}

frame IC,Adv0,Adv,AdvD,Ur,Ui,Crank,CrankR,Urout,Diff,UIout,psi,vel;

box(IC,cell("initial condition $\v U_0$",(0,1)),margin,black,
    FillDraw(palegray));
ellipse(Adv0,cell("Lagrangian state $\v U(t)$",(1,1)),margin,red,
	FillDraw(palered));
ellipse(Adv,cell("Lagrangian prediction $\v U(t+\tau)$",(1,0)),margin,red,
	FillDraw(palered));
ellipse(AdvD,cell("diffused parcels",(1.8,1)),margin,red,FillDraw(palered));
box(Ur,cell("rearranged $\v \widetilde U$",(0,0)),margin,orange+gray,
    FillDraw(paleyellow));
box(Ui,cell("interpolated $\v \widetilde U$",(1,-1)),margin,blue,
    FillDraw(paleblue));
box(Crank,cell("${\cal L}^{-1}(-\tau){\cal L}(\tau)\v \widetilde U$",(0.5,-1)),
    margin,blue,FillDraw(paleblue));
box(CrankR,cell("${\cal L}^{-1}(-\tau){\cal L}(\tau)\v \widetilde U$",
		(0,-1)),margin,orange+gray,
    FillDraw(paleyellow));
box(Urout,cell(minipage("\center{Lagrangian rearranged solution~$\v U_R$}",
			100pt),(0,-2)),margin,orange+gray,FillDraw(paleyellow));
box(Diff,cell("$\v D\del^2 \v \widetilde U$",(0.75,-1.5)),margin,blue,
    FillDraw(paleblue));
box(UIout,cell(minipage("\center{semi-Lagrangian solution~$\v U_I$}",80pt),
	       (0.5,-2)),margin,FillDraw(palered+paleyellow));
box(psi,cell("$\psi=\del^{-2}\omega$",(1.6,-1)),margin,darkgreen,
    FillDraw(palegreen));
box(vel,cell("$\v v=\v{\hat z} \cross\grad\psi$",(1.6,-0.5)),margin,darkgreen,
    FillDraw(palegreen));

pair padv=0.5*(point(Adv0,S)+point(Adv,N));

add(IC);
add(Adv0);
add(Adv);
add(AdvD);
add(Ur);
add(Ui);
add(Crank);
add(CrankR);
add(Urout);
add(Diff);
add(UIout);
add(psi);
add(vel);

draw("initialize",point(IC,E)--point(Adv0,W),RightSide,Arrow,PenMargin);
draw(minipage("\flushright{advect: Runge-Kutta}",80pt),
     point(Adv0,S)--point(Adv,N),RightSide,red,Arrow,PenMargin);
draw(Label("Lagrange $\rightarrow$ Euler",0.45),point(Adv,W)--point(Ur,E),LeftSide,orange+gray,
     Arrow,PenMargin);
draw("Lagrange $\rightarrow$ Euler",point(Adv,S)--point(Ui,N),LeftSide,blue,
     Arrow,PenMargin);
draw(point(Adv,E)--(point(AdvD,S).x,point(Adv,E).y),red,Arrow(Relative(0.7)),
     PenMargin);
draw(minipage("\flushleft{diffuse: multigrid Crank--Nicholson}",80pt),
     point(Ui,W)--point(Crank,E),5N,blue,MidArrow,PenMargin);
draw(minipage("\flushleft{diffuse: multigrid Crank--Nicholson}",80pt),
     point(Ur,S)--point(CrankR,N),LeftSide,orange+gray,Arrow,PenMargin);
draw("output",point(CrankR,S)--point(Urout,N),RightSide,orange+gray,Arrow,PenMargin);
draw(point(Ui,S)--point(Diff,N),blue,MidArrow,PenMargin);
draw(point(Crank,S)--point(Diff,N),blue,MidArrow,PenMargin);
label("subtract",point(Diff,N),6N,blue);
draw(Label("Euler $\rightarrow$ Lagrange",0.5),
     point(Diff,E)--(point(AdvD,S).x,point(Diff,E).y)--
     (point(AdvD,S).x,point(Adv,E).y),RightSide,blue,Arrow(position=1.5),
     PenMargin);
dot((point(AdvD,S).x,point(Adv,E).y),red);
draw((point(AdvD,S).x,point(Adv,E).y)--point(AdvD,S),red,Arrow,PenMargin);
draw("output",point(Crank,S)--point(UIout,N),RightSide,brown,Arrow,PenMargin);
draw(Label("$t+\tau\rightarrow t$",0.45),point(AdvD,W)--point(Adv0,E),LeftSide,red,Arrow,
PenMargin);
draw(point(psi,N)--point(vel,S),darkgreen,Arrow,PenMargin);
draw(Label("self-advection",5.5),point(vel,N)--
     arc((point(vel,N).x,point(Adv,E).y),5,270,90)--(point(vel,N).x,padv.y)--
     padv,LeftSide,darkgreen,Arrow,PenMargin);
draw(Label("multigrid",0.5,S),point(Ui,E)--point(psi,W),darkgreen,
     Arrow,PenMargin);

shipout(Landscape);

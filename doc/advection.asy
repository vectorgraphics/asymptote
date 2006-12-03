size(0,22cm);

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

object IC,Adv0,Adv,AdvD,Ur,Ui,Crank,CrankR,Urout,Diff,UIout,psi,vel;

IC=object(box,Label("initial condition $\v U_0$",(0,1)),
	  margin,black,FillDraw(palegray));
Adv0=object(ellipse,Label("Lagrangian state $\v U(t)$",(1,1)),
	    margin,red,FillDraw(palered));
Adv=object(ellipse,Label("Lagrangian prediction $\v U(t+\tau)$",(1,0)),
	   margin,red,FillDraw(palered));
AdvD=object(ellipse,Label("diffused parcels",(1.8,1)),
	    margin,red,FillDraw(palered));
Ur=object(box,Label("rearranged $\v \widetilde U$",(0,0)),
	  margin,orange+gray,FillDraw(paleyellow));
Ui=object(box,Label("interpolated $\v \widetilde U$",(1,-1)),
	  margin,blue,FillDraw(paleblue));
Crank=object(box,Label("${\cal L}^{-1}(-\tau){\cal L}(\tau)\v \widetilde U$",
		       (0.5,-1)),margin,blue,FillDraw(paleblue));
CrankR=object(box,Label("${\cal L}^{-1}(-\tau){\cal L}(\tau)\v \widetilde U$",
			(0,-1)),margin,orange+gray,FillDraw(paleyellow));
Urout=object(box,
	     Label(minipage("\center{Lagrangian rearranged solution~$\v U_R$}",
			    100pt),
		   (0,-2)),margin,orange+gray,FillDraw(paleyellow));
Diff=object(box,Label("$\v D\del^2 \v \widetilde U$",(0.75,-1.5)),
	    margin,blue,FillDraw(paleblue));
UIout=object(box,Label(minipage("\center{semi-Lagrangian solution~$\v U_I$}",
				80pt),
		       (0.5,-2)),margin,FillDraw(palered+paleyellow));
psi=object(box,Label("$\psi=\del^{-2}\omega$",(1.6,-1)),
	   margin,darkgreen,FillDraw(palegreen));
vel=object(box,Label("$\v v=\v{\hat z} \cross\grad\psi$",(1.6,-0.5)),
	   margin,darkgreen,FillDraw(palegreen));

add(new void(frame f, transform t) {
    pair padv=0.5*(point(Adv0,S,t)+point(Adv,N,t));
    picture pic;
    draw(pic,"initialize",point(IC,E,t)--point(Adv0,W,t),RightSide,Arrow,
	 PenMargin);
    draw(pic,minipage("\flushright{advect: Runge-Kutta}",80pt),
	 point(Adv0,S,t)--point(Adv,N,t),RightSide,red,Arrow,PenMargin);
    draw(pic,Label("Lagrange $\rightarrow$ Euler",0.45),
	 point(Adv,W,t)--point(Ur,E,t),5LeftSide,orange+gray,
	 Arrow,PenMargin);
    draw(pic,"Lagrange $\rightarrow$ Euler",point(Adv,S,t)--point(Ui,N,t),
	 RightSide,blue,Arrow,PenMargin);
    draw(pic,point(Adv,E,t)--(point(AdvD,S,t).x,point(Adv,E,t).y),red,
	 Arrow(Relative(0.7)),PenMargin);
    draw(pic,minipage("\flushleft{diffuse: multigrid Crank--Nicholson}",80pt),
	 point(Ui,W,t)--point(Crank,E,t),5N,blue,MidArrow,PenMargin);
    draw(pic,minipage("\flushleft{diffuse: multigrid Crank--Nicholson}",80pt),
	 point(Ur,S,t)--point(CrankR,N,t),LeftSide,orange+gray,Arrow,PenMargin);
    draw(pic,"output",point(CrankR,S,t)--point(Urout,N,t),RightSide,
	 orange+gray,Arrow,PenMargin);
    draw(pic,point(Ui,S,t)--point(Diff,N,t),blue,MidArrow,PenMargin);
    draw(pic,point(Crank,S,t)--point(Diff,N,t),blue,MidArrow,PenMargin);
    label(pic,"subtract",point(Diff,N,t),12N,blue);
    draw(pic,Label("Euler $\rightarrow$ Lagrange",0.5),
	 point(Diff,E,t)--(point(AdvD,S,t).x,point(Diff,E,t).y)--
	 (point(AdvD,S,t).x,point(Adv,E,t).y),RightSide,blue,
	 Arrow(position=1.5),PenMargin);
    dot(pic,(point(AdvD,S,t).x,point(Adv,E,t).y),red);
    draw(pic,(point(AdvD,S,t).x,point(Adv,E,t).y)--point(AdvD,S,t),red,Arrow,
	 PenMargin);
    draw(pic,"output",point(Crank,S,t)--point(UIout,N,t),RightSide,brown,Arrow,
	 PenMargin);
    draw(pic,Label("$t+\tau\rightarrow t$",0.45),
	 point(AdvD,W,t)--point(Adv0,E,t),2.5LeftSide,red,Arrow,PenMargin);
    draw(pic,point(psi,N,t)--point(vel,S,t),darkgreen,Arrow,PenMargin);
    draw(pic,Label("self-advection",5.5),point(vel,N,t)--
	 arc((point(vel,N,t).x,point(Adv,E,t).y),5,270,90)--
	 (point(vel,N,t).x,padv.y)--
	 padv,LeftSide,darkgreen,Arrow,PenMargin);
    draw(pic,Label("multigrid",0.5,S),point(Ui,E,t)--point(psi,W,t),darkgreen,
	 Arrow,PenMargin);

    add(f,pic.fit());
  });

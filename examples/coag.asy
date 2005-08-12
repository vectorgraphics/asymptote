size(0,200);
import graph;

pair z0=(0,0);
pair m0=(0,1);
pair tg=(1.5,0);
pair mt=m0+tg;
pair tf=(3,0);

draw(m0--mt{dir(-70)}..{dir(0)}2tg+m0/4);
xtick("$T_g$",tg,N);
label("$M(t)$",mt,2NE);
labely("$M_0$",m0);

xaxis(Label("$t$",align=2S),Arrow);
yaxis(Arrow);

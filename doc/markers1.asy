size(12cm,0);
import markers;

pair A=(0,0), B=(1,0), C=(2,0), D=(3,0);
path p=A--B--C--D;
transform T=shift(-4,-1);
transform t=shift(4,0);

//line 1 **********
draw(p,marker(markinterval(3,dotframe,true)));
label("$1$",point(p,0),3W);

//line 2 **********
p=t*p;
draw(p,marker(stickframe,markuniform(4)));
label("$2$",point(p,0),3W);

//line 3 **********
p=T*p;
draw(p,marker(stickframe(red),markinterval(3,dotframe(blue),true)));
label("$3$",point(p,0),3W);

//line 4 **********
p=t*p;
draw(p,marker(dotframe(red),markinterval(3,stickframe(2,blue),true)));
label("$4$",point(p,0),3W);

//line 5 **********
p=T*p;
pen pn=linewidth(4bp);
draw(p,pn,marker(dotframe(red+pn),markinterval(3,stickframe(3,angle=25,pn),
					      true)));
label("$5$",point(p,0),3W);

//line 6 **********
p=t*p;
draw(p,marker(scale(2)*dotframe(red),
	      markinterval(3,stickframe(5,angle=25,size=4mm,space=2mm,
				       offset=I*2mm),true)));
label("$6$",point(p,0),3W);

//line 7 **********
p=T*p;
draw(p,marker(dotframe,
	      markinterval(2,stickframe(3,angle=45,space=3mm,size=10mm),true)));
label("$7$",point(p,0),3W);

//line 8 **********
p=t*p;
draw(p,marker(dotframe,markinterval(2,circlebarframe(2),true)));
label("$8$",point(p,0),3W);

//line 9 **********
p=T*p;
draw(p,marker(dotframe,
	      markinterval(2,circlebarframe(3,angle=30,barsize=8mm,
					   radius=2mm,FillDraw(.8red)),true)));
label("$9$",point(p,0),3W);

//line 10 **********
p=t*p;
draw(p,marker(dotframe,
	      markinterval(2,circlebarframe(3,angle=30,barsize=8mm,
					   radius=2mm,FillDraw(.8red),
					   above=true),true)));
label("$10$",point(p,0),3W);

//line 11 **********
p=T*p;
draw(p,marker(dotframe,markinterval(2,circlebarframe(3,angle=30,barsize=8mm,
						    radius=2mm,FillDraw(.8red),
						    above=true),true),put=Below));
label("$11$",point(p,0),3W);

//line 12 **********
p=t*p;
draw(p,marker(dotframe,markinterval(3,tildeframe,true)));
label("$12$",point(p,0),3W);

//line 13 **********
p=T*p;
draw(p,marker(dotframe,markinterval(3,tildeframe(2,angle=-20),true)));
label("$13$",point(p,0),3W);

//line 14 **********
p=t*p;
draw(p,marker(dotframe,markinterval(3,crossframe(3),true)));
label("$14$",point(p,0),3W);

//line 15 **********
p=shift(.25S)*T*p;
path cle=shift(relpoint(p,.5))*scale(abs(A-D)/4)*unitcircle;
draw(cle,marker(dotframe(6bp+red),markinterval(5,stickframe(3),true)));
label("$15$",point(p,0),3W);

//line 16 **********
cle=t*cle;
p=t*p;
frame a;
label(a,"$a$",(0,-2labelmargin()));
draw(cle,marker(dotframe(6bp+red),markinterval(5,a,true)));
label("$16$",point(p,0),3W);

//line 17 **********
p=T*shift(relpoint(p,.5)+.65S)*scale(.5)*shift(-relpoint(p,.5))*
  rotate(45,relpoint(p,.5))*p;
draw(p,marker(dotframe,markinterval(2,tildeframe(size=5mm))));
label("$17$",point(p,0),3W);

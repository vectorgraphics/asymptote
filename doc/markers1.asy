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
draw(p,StickIntervalMarker(3,2,blue,dotframe(red)));
label("$4$",point(p,0),3W);

//line 5 **********
p=T*p;
pen pn=linewidth(4bp);
draw(p,pn,StickIntervalMarker(3,3,angle=25,pn,dotframe(red+pn)));
label("$5$",point(p,0),3W);

//line 6 **********
p=t*p;
draw(p,StickIntervalMarker(3,5,angle=25,size=4mm,space=2mm,offset=I*2mm,
                           scale(2)*dotframe(red)));
label("$6$",point(p,0),3W);

//line 7 **********
p=T*p;
draw(p,StickIntervalMarker(n=3,angle=45,size=10mm,space=3mm,dotframe));
label("$7$",point(p,0),3W);

//line 8 **********
p=t*p;
draw(p,CircleBarIntervalMarker(n=2,dotframe));
label("$8$",point(p,0),3W);

//line 9 **********
p=T*p;
draw(p,CircleBarIntervalMarker(n=3,angle=30,barsize=8mm,radius=2mm,
                               FillDraw(.8red),
                               dotframe));
label("$9$",point(p,0),3W);

//line 10 **********
p=t*p;
draw(p,CircleBarIntervalMarker(n=3,angle=30,barsize=8mm,radius=2mm,
                               FillDraw(.8red),circleabove=true,dotframe));
label("$10$",point(p,0),3W);

//line 11 **********
p=T*p;
draw(p,CircleBarIntervalMarker(n=3,angle=30,barsize=8mm,radius=2mm,
                               FillDraw(.8red),circleabove=true,dotframe,
			       above=false));
label("$11$",point(p,0),3W);

//line 12 **********
p=t*p;
draw(p,TildeIntervalMarker(i=3,dotframe));
label("$12$",point(p,0),3W);

//line 13 **********
p=T*p;
draw(p,TildeIntervalMarker(i=3,n=2,angle=-20,dotframe));
label("$13$",point(p,0),3W);

//line 14 **********
p=t*p;
draw(p,CrossIntervalMarker(3,3,dotframe));
label("$14$",point(p,0),3W);

//line 15 **********
p=shift(.25S)*T*p;
path cle=shift(relpoint(p,.5))*scale(abs(A-D)/4)*unitcircle;
draw(cle,StickIntervalMarker(5,3,dotframe(6bp+red)));
label("$15$",point(p,0),3W);

//line 16 **********
cle=t*cle;
p=t*p;
frame a;
label(a,"$a$",(0,-2labelmargin()));
draw(cle,marker(dotframe(6bp+red),markinterval(5,a,true)));
label("$16$",point(p,0),3W);

// line 17 **********
p=T*shift(relpoint(p,.5)+.65S)*scale(.5)*shift(-relpoint(p,.5))*rotate(45,relpoint(p,.5))*p;
draw(p,TildeIntervalMarker(size=5mm,rotated=false,dotframe));
label("$17$",point(p,0),3W);

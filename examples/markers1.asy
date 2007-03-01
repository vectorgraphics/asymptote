size(15cm,0);
import markers;

pair A=(0,0), B=(1,0), C=(2,0), D=(3,0);
path p=A--B--C--D;
transform T=shift(-4,-1);
transform t=shift(4,0);

//line 1 **********
draw(p,markersuniform(4,newframe,dotframe));
label("$1$",point(p,0),3W);

//line 2 **********
p=t*p;
draw(p,markersuniform(4,stickframe,newframe));
label("$2$",point(p,0),3W);

//line 3 **********
p=T*p;
draw(p,markersuniform(4,stickframe(red),dotframe(blue)));
label("$3$",point(p,0),3W);

//line 4 **********
p=t*p;
draw(p,markersuniform(4,dotframe(red),stickframe(2,blue)));
label("$4$",point(p,0),3W);

//line 5 **********
p=T*p;
pen pn=linewidth(4bp);
draw(p,pn,markersuniform(4,dotframe(red+pn),stickframe(3,angle=25,pn)));
label("$5$",point(p,0),3W);

//line 6 **********
p=t*p;
draw(p,markersuniform(4,scale(2)*dotframe(red),stickframe(5,angle=25,size=4mm,space=2mm,voffset=2mm)));
label("$6$",point(p,0),3W);

//line 7 **********
p=T*p;
draw(p,markersuniform(3,dotframe,stickframe(3,angle=45,space=3mm,size=10mm)));
label("$7$",point(p,0),3W);

//line 8 **********
p=t*p;
draw(p,markersuniform(3,dotframe,circlebarframe(2)));
label("$8$",point(p,0),3W);

//line 9 **********
p=T*p;
draw(p,markersuniform(3,dotframe,circlebarframe(3,angle=30,barsize=8mm,radius=2mm,FillDraw(.8red))));
label("$9$",point(p,0),3W);

//line 10 **********
p=t*p;
draw(p,markersuniform(3,dotframe,circlebarframe(3,angle=30,barsize=8mm,radius=2mm,FillDraw(.8red),above=true)));
label("$10$",point(p,0),3W);

//line 11 **********
p=T*p;
draw(p,markersuniform(3,dotframe,circlebarframe(3,angle=30,barsize=8mm,radius=2mm,FillDraw(.8red),above=true),put=Below));
label("$11$",point(p,0),3W);

//line 12 **********
p=t*p;
draw(p,markersuniform(4,dotframe,tildeframe));
label("$12$",point(p,0),3W);

//line 13 **********
p=T*p;
draw(p,markersuniform(4,dotframe,tildeframe(2,angle=-20)));
label("$13$",point(p,0),3W);

//line 14 **********
p=t*p;
draw(p,markersuniform(4,dotframe,crossframe(3)));
label("$14$",point(p,0),3W);

//line 15 **********
p=shift(.25S)*T*p;
path cle=shift(relpoint(p,.5))*scale(abs(A-D)/4)*unitcircle;
draw(cle,markersuniform(6,dotframe(6bp+red),stickframe(3)));
label("$15$",point(p,0),3W);

//line 16 **********
cle=t*cle;
p=t*p;
picture a;
label(a,"$a$",(0,-2labelmargin()));
draw(cle,markersuniform(6,dotframe(6bp+red),a.fit()));
label("$16$",point(p,0),3W);

//line 17 **********
p=T*shift(relpoint(p,.5)+.65S)*scale(.5)*shift(-relpoint(p,.5))*rotate(45,relpoint(p,.5))*p;
draw(p,markersuniform(3,dotframe,tildeframe(size=5mm),rotated=false));
label("$17$",point(p,0),3W);

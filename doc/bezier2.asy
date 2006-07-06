size(400);
pair p1,p2,p3,p4;
p1=(0,0);
p2=(1,1);
p3=(2,1);
p4=(3,0);
draw(p1..controls p2 and p3 .. p4,blue+dashed); //bezier curve

draw(p1--p2--p3--p4);
dot("p1",p1,red);
dot("p2",p2,red);
dot("p3",p3,red);
dot("p4",p4,red);

pair midpoint(pair a, pair b){return (a+b)*0.5;}
pair p5=midpoint(p1,p2);
pair p6=midpoint(p2,p3);
pair p7=midpoint(p3,p4);

draw(p5--p6--p7);
dot("p5",p5,red);
dot("p6",p6,red);
dot("p7",p7,red);

pair p8=midpoint(p5,p6);
pair p9=midpoint(p6,p7);
pair p10=midpoint(p8,p9);

draw(p8--p9);
dot("p8",p8,red);
dot("p9",p9,red);
dot("p10",p10,red);

import three;
import graph3;
import barGraph;

size(300);

//Examples of 3D bargraphs!!!

//currentlight=White; will remove and stuff 
light Headlamp=light(gray(0.8),specular=gray(0.7),specularfactor=3,dir(8,-90));
currentlight=Headlamp;
//currentprojection=perspective(30,30,30,up=Z);
currentprojection=perspective(-30,-30,30,up=Z);//will remove, for testing reasons

//BARGRAPH3 1
//9 bars organized in 3 by 3 rows, each with a different color (the colors match one to one with the heights,
// and match the heights as readd left to right)
//x and y axis labels 
real[][] h1={{2,3,4},{2,4,5},{6,7,8}};
pen[] colors1={pink,blue,purple,magenta,olive,lightblue,lightolive,palecyan,paleyellow};
string[] xl={"This","sss","uuuppp"};
string[] yl={"queen", "now", "IIII"};
barGraph3(width=2,h1,colors1,ylabels=yl,xl);


//BARGRAPH3 2
//3 bars organized in one row along the x-axis
real[][] h2={{2},{5},{6}};
pen[] colors2={pink,purple,blue};//same output if pen[][] {{colors in here}}
string[] xl={"bars","for","days"};
string[] yl={"1 row"};
//barGraph3(width=1,h2,colors2,ylabels=yl,xl);


//BARGRAPH3 3
//3 bars organized in one row along the y-axis
real[][] h3={{3,4,6}};//do to casting real[] {} and real[][] {{},{},{}} would not look the same
pen[] colors3={darkolive,olive,paleyellow};//same output if pen[][] {{colors in here}}
string[] xl={"1 row"};
string[] yl={"new", "now", "knew"};
//barGraph3(width=2,h3,colors3,ylabels=yl,xl);




xaxis3(Bounds);//"$x$",Bounds,InTicks(endlabel=false,2,2));
yaxis3(Bounds);//"$y$",Bounds,InTicks(beginlabel=false,2,2));
zaxis3("$z$",Bounds,InTicks);



//pen[] colours={yellow,blue,green,red,lightblue,magenta,olive,pink};
//colours.cyclic=true;
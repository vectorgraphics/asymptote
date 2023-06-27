import barGraph;
import graph;

size(400,y=300, false);

real[] h={12,21,7,15};
pen[] p={pink, blue, purple, magenta+0.5lightblue};
string[] l={"pens","bar 2","last","v last"};
pen[] pp={black};


//barGraph(width=2, height = h,labels=l,drawpens=purple, fillpens=p,labelpens=blue); // figure out defaults with currentpen and stuff 
/*
xaxis(-.5,RightTicks(n=1));
yaxis(-.25,4*4+2,above=true);
*/

filltype[] ft={AxialShade(olive+0.15palegreen,darkolive), AxialShade(paleblue,lightblue)};
//barGraph(width=2,height=h,labels=l,ft,labelpens=new pen[] {darkbrown});

real[] h2={1,2,3,4,5,6,7,8,9,10,11};
real[] h3;
for (int i=0;i<h2.length;++i){
    h3.push(h2[i]*0.5);
}
string[] l2={"","pens","","","","bar 2","","","", "last",""};
filltype[] ft2={FillDraw(deepblue,currentpen),FillDraw(darkolive,currentpen),FillDraw(darkmagenta,currentpen),NoFill};
filltype[] ft3={FillDraw(lightblue,currentpen),FillDraw(olive,currentpen),FillDraw(pink,currentpen),NoFill};
string[] l3={""};
l3.cyclic=true;
//barGraph(width=3,distance=0.5,height=h2,ft3,labels=l2,horizontal);
//barGraph(width=3,distance=0.5,height=h3,ft2,labels=l3,horizontal);

real[] h4={-1,-2,-3,4,5,6,7};
string[] l4={"","pens","","","","bar 2",""};
barGraph(width=2,distance=0,height=h4,ft3,l4);

xaxis(-.5,2*10,above=true);
yaxis(-.5,LeftTicks(n=0));
/*
xaxis(-.5,RightTicks(n=1));
yaxis(-.25,4*4+2,above=true);
*/
import barGraph;
import graph;

size(450,y=400, false);


//BARGRAPH 1 
//4 columns, 4 drawing colors but will not be filled, 4 labels
real[] h1={12,21,7,15};
filltype[] ft1={FillDraw(pink),Draw(blue),Draw(purple),Draw(magenta+0.5lightblue)};
string[] l1={"pens","bar 2","last","v last"};
//barGraph(width=2,height=h1,distance=0.5,ft1,l1); 


//BARGRAPH 2 
//4 columns, each with 2 or 3 layers, the filltypes are in a nested array, 4 labels
real[][] h2={{12,9},{7,10,5},{4,7,2},{13,5}};
string[] l2={"pens","bar 2","last","v last"};
pen pen1=currentpen;
filltype[][] ft2={{FillDraw(darkolive,pen1),FillDraw(olive,pen1)},
        {FillDraw(magenta+0.6darkblue,pen1),FillDraw(magenta+0.5white,pen1),FillDraw(pink,pen1)},
        {FillDraw(darkblue,pen1),FillDraw(blue,pen1),FillDraw(lightblue,pen1)},
        {FillDraw(brown,pen1),FillDraw(heavyred,pen1)}};
//barGraph(width=2,distance=0.5,height=h2,ft2,l2,clumps=2);


//BARGRAPH 3
//6 columns, each with 2 or 3 layers, yet the filltypes are in a simple array (get repeated each column)
real[][] h3={{91,82},{68,45},{93,56,59},{32,57},{63,26,59},{73,52}};
string[] l3={"","bars 1","","","bars 2",""};
filltype[] ft3={FillDraw(lightblue,currentpen),FillDraw(olive,currentpen),FillDraw(pink,currentpen)};
barGraph(width=3,distance=0.3,height=h3,labels=l3,filltypes=ft3,clumps=3);

 
//BARGRAPH 4
//Horizontal bar graph with 3 columns and 2 to 3 layers per column
real[][] h44={{1,3.4,2},{3,4,1},{5,6}};
string[] l4={"1","pens","last","bar","bar","2"};
filltype[] ft4={FillDraw(deepcyan,currentpen),FillDraw(cyan,currentpen),FillDraw(palecyan,currentpen)};
//barGraph(width=2,height=h44,ft4,labels=l4,horizontal);

//THE AXIS
/*xaxis(-.5,above=true);
yaxis(-.5,LeftTicks(n=0));
*/
xaxis(-.5,14,above=true);
yaxis(-.25,LeftTicks(n=0));
/*
xaxis(BottomTop(true),LeftTicks);
yaxis(LeftRight);

*/

//THE ALLIES


import piechart;
size(300);

//Examples for piecharts!!!


//FIX: the percent and inside label pls


//PIECHART 1
//regular pie chart with percent and legend
pair a1=(0,0);
real r1=15;
real[] h1={20,10,24,16,19,11};
pen[] p1={paleblue,paleblue+0.5lightblue,lightblue,lightblue+0.5mediumblue,
    mediumblue,mediumblue+0.5blue,blue,heavyblue,deepblue};
//string[] l1={"I","did","not","commit","war","crime"};
Label[] l1={Label("I"),Label("did"),Label("not"),Label("commit"),Label("war"),Label("crime")};
piechart(a1,r1,h1,pens=p1,labels=l1,outside);


//PIECHART 2
//pie chart with some elements outside and a legend
pair c2=(20,20);
real r2=20;
real[] h2={20,10,9,11,15,9,6,12,8};//};
pen[] p2={pink,olive,lightblue,magenta,purple,blue,darkolive,brown,cyan,heavymagenta};
string[] l2={"123","smh","not","dark","purple","one","!","other","another","else"};
int[] d2={1,0,0,1,2,0,3,0,1,0};
//piechart(c2,radius=r2,h2,p2,percent,l2,labeltype=outside,d2);

//PIECHART 3
//this pie chart the labels are angled 
pair c3=(20,20);
real r3=10;
real[] h3={20,10,20,15,15,12,8};//};
pen[] p3={pink,olive,lightblue,magenta,purple,blue,darkolive,brown,cyan,heavymagenta};
Label[] l3={rotate(-70)*Label("I"),Label("did"),Label("not"),Label("commit"),Label("war"),Label("crime"),Label("!!!!!")};
//I'll have to think about how to rotate the labels and what angles 
int[] d3={1,0,0,1,2,0,3,0,1,0};
//piechart(c3,r3,h3,p3,legend1,l3);




/*
draw((0,0)--(0,30));
draw((0,0)--(30,0));    */
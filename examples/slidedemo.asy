// Slide demo.
// Command-line options to enable stepping and/or reverse video:
// asy [-u stepping=true] [-u reverse=true] [-u itemstep=true] slidedemo

orientation=Landscape;

import slide;
import three;

viewportsize=pagewidth-2pagemargin;

usersetting();

// Commands to generate optional bibtex citations:
// asy slidedemo
// bibtex slidedemo_
// asy slidedemo
//
bibliographystyle("alpha");

// Generated needed files if they don't already exist.
asy(nativeformat(),"Pythagoras","log","PythagoreanTree");
usepackage("mflogo");

// Optional background color or header:
// import x11colors;
// fill(background,box((-1,-1),(1,1)),Azure);
// label(background,"Header",(0,startposition.y));

titlepage(title="Slides with {\tt Asymptote}: A Demo",
          author="John C. Bowman",
          institution="University of Alberta",
          date="\today",
          url="http://asymptote.sf.net");

outline("Basic Commands");
item("item");
subitem("subitem");
remark("remark");
item("draw \cite{Hobby86,Knuth86b}");
item("figure");
item("embedded and external animations: see {\tt slidemovie.asy}");

title("Items");
item("First item.");
subitem("First subitem.");
subitem("Second subitem.");
item("Second item.");
equation("a^2+b^2=c^2.");
equations("\frac{\sin^2\theta+\cos^2\theta}{\cos^2\theta}
&=&\frac{1}{\cos^2\theta}\nonumber\\
&=&\sec^2\theta.");
remark("A remark.");
item("To enable pausing between bullets:");
remark("{\tt asy -u stepping=true}");
item("To enable reverse video:");
remark("{\tt asy -u reverse=true}");

title("Can draw on a slide, preserving the aspect ratio:");
picture pic,pic2;
draw(pic,unitcircle);
add(pic.fit(15cm));
step();
fill(pic2,unitcircle,paleblue);
label(pic2,"$\pi$",(0,0),fontsize(500pt));
add(pic2.fit(15cm));

newslide();
item("The slide \Red{title} \Green{can} \Blue{be} omitted.");
figure("Pythagoras","height=12cm",
       "A simple proof of Pythagoras' Theorem.");

newslide();
item("Single skip:");
skip();
item("Double skip:");
skip(2);
figure(new string[] {"log."+nativeformat(),"PythagoreanTree."+nativeformat()},
       "width=10cm",new string[] {"{\tt log.asy}","{\tt PythagoreanTree.asy}"},
       "Examples of {\tt Asymptote} output.");

title("Embedded Interactive 3D Graphics");
picture pic;
import graph3;
import solids;
viewportmargin=(0,1cm);
currentprojection=orthographic(1,0,10,up=Y);
pen color=green;
real alpha=-240;
real f(real x) {return sqrt(x);}
pair F(real x) {return (x,f(x));}
triple F3(real x) {return (x,f(x),0);}
path p=graph(pic,F,0,1,n=30,operator ..)--(1,0)--cycle;
path3 p3=path3(p);
revolution a=revolution(p3,X,alpha,0);
render render=render(compression=0,merge=true);
draw(pic,surface(a),color,render);
draw(pic,p3,blue);
surface s=surface(p);
draw(pic,s,color,render);
draw(pic,rotate(alpha,X)*s,color,render);
xaxis3(pic,Label("$x$",1),xmax=1.25,dashed,Arrow3);
yaxis3(pic,Label("$y$",1),Arrow3);
dot(pic,"$(1,1)$",(1,1,0));
arrow(pic,"$y=\sqrt{x}$",F3(0.8),Y,0.75cm,red);
real r=0.4;
draw(pic,F3(r)--(1,f(r),0),red);
real x=(1+r)/2;
draw(pic,"$r$",(x,0,0)--(x,f(r),0),X+0.2Z,red,Arrow3);
draw(pic,arc(1.1X,0.4,90,90,3,-90),Arrow3);
add(pic.fit(0,14cm));

title("\mbox{Asymptote: 2D \& 3D Vector Graphics Language}");
asyinclude("logo3");
center("\tt http://asymptote.sf.net");
center("(freely available under the LGPL license)");

bibliography("refs");

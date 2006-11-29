// Slide demo.
// Command-line options to enable stepping and/or reverse video:
// asy [-u stepping=true] [-u reverse=true] slidedemo

orientation=Landscape;

settings.tex="pdflatex";

// Generated needed files if they don't already exist.
asy(nativeformat(),"Pythagoras","log","near_earth");
asy("mpg","animations/wheel");

import slide;

// Optional movie modules:
import pdfanim;     // For portable embedded PDF movies (version 0.53 or later)
import external;    // For portable external movies
import embed;       // For non-portable embedded movies

usersetting();

// Optional background color:
// import x11colors;
// fill(background,box((-1,-1),(1,1)),Azure);

titlepage("Slides with {\tt Asymptote}: A Demo","John C. Bowman",
	  "University of Alberta","\today","http://asymptote.sf.net");

outline();
item("item");
subitem("subitem");
remark("remark");
item("draw");
item("figure");

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
figure(new string[] {"log."+nativeformat(),"near_earth."+nativeformat()},
       "width=10cm",new string[] {"{\tt log.asy}","{\tt near\_earth.asy}"},
       "Examples of {\tt Asymptote} graphs.");

title("Embedded PDF movies (portable)");
animation a=animation("A");
animation b=animation("B");
int n=20;
for(int i=0; i < 2n; ++i) {
  picture pic;
  size(pic,100);
  draw(pic,shift(0,sin(pi/n*i))*unitsquare);
  a.add(pic);
  if(i < 1.5n) b.add(rotate(45)*pic);
}
display(new string[] {a.pdf("remember,auto,loop"),b.pdf()},
	new string[] {baseline(a.controlpanel(percentage=true)),
	    b.controlpanel()+b.progress(blue)},
  "Click on animations to Play/Pause/Reset;\\Shift click to Reverse");

title("External Movie (portable)");
display(external.embed("animations/wheel.mpg",
		       "poster,text=wheel.mpg,label=wheel.mpg",20cm,5.6cm));
display(external.link("animations/wheel.mpg","Play","play"));

title("Embedded Movie (not portable)");
display(embed.embed("animations/wheel.mpg",
		   "poster,text=wheel.mpg,label=wheel.mpg",
		   20cm,5.6cm));
display(embed.link("animations/wheel.mpg","Play","play"));

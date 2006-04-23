orientation=Landscape;

import slide;
import x11colors;

fill(background,box((-1,-1),(1,1)),Azure);

stepping=true;

titlepage("Slides with {\tt Asymptote}: A Demo","John C. Bowman","\today",
	  "http://asymptote.sf.net");

title("Slide title");

item("First item.");
remark("A remark.");
item("Second item.");
equation("c^2=a^2+b^2.");
item("Third item.");
subitem("First subitem.");
subitem("Second subitem.");
item("Set {\tt stepping=false} for posting on WWW or printing.");
newslide();

item("The slide title can be omitted.");
figure("Pythagoras.eps","height=12cm","A simple proof of Pythagoras' Theorem.");
newslide();

item("Single skip:");
skip();
item("Double skip:");
skip(2);
figure(new string[] {"tan.eps","near_earth.eps"},
       "height=12cm","Examples of {\tt Asymptote} graphs.");
newslide();

title("Can draw on a slide, preserving the aspect ratio:");
picture pic,pic2;
draw(pic,unitcircle);
add(pic.fit(15cm));
step();
fill(pic2,unitcircle,paleblue);
add(pic2.fit(15cm));

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
figure("logo.eps","height=12cm","The {\tt Asymptote} logo.");
newslide();

item("Single skip:");
skip();
item("Double skip:");
skip(2);
figure(new string[] {"loggraph.eps","lineargraph.eps"},
       "height=8cm","Examples of {\tt Asymptote} graphs.");
newslide();

item("It's easy to draw on a slide, reserving the aspect ratio:");
step();
picture pic;
draw(pic,unitcircle);
add(pic.fit(15cm));

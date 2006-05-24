// Slide demo.

orientation=Landscape;

// Generated needed files if they don't already exist.
asy(overwrite=false,"Pythagoras","log","near_earth");

import slide;
import x11colors;

// Allow user to enable stepping and/or reverse video:
// asy -u stepping=true -u reverse=true

usersetting();

// Optional background color:
// fill(background,box((-1,-1),(1,1)),Azure);

titlepage("Slides with {\tt Asymptote}: A Demo","John C. Bowman","\today",
	  "http://asymptote.sf.net");

title("Outline");
item("item");
subitem("subitem");
remark("remark");
item("draw");
item("figure");

newslide(stepping=false);
title("Outline",newslide=false);
item("First item.");
remark("A remark.");
item("Second item.");
equation("c^2=a^2+b^2.");
item("Third item.");
subitem("First subitem.");
subitem("Second subitem.");
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
item("The slide \Red{title} can be omitted.");
figure("Pythagoras.eps","height=12cm",
       "A simple proof of Pythagoras' Theorem.");

newslide();
item("Single skip:");
skip();
item("Double skip:");
skip(2);
figure(new string[] {"log.eps","near_earth.eps"},
       "width=10cm","Examples of {\tt Asymptote} graphs.");


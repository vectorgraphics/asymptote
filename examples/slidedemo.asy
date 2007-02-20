// Slide demo.
// Command-line options to enable stepping and/or reverse video:
// asy [-u stepping=true] [-u reverse=true] slidedemo

orientation=Landscape;

// Generated needed files if they don't already exist.
asy(nativeformat(),"Pythagoras","log","near_earth");

// Commands to generate optional bibtex citations:
// asy -k slidedemo
// bibtex slidedemo_
// asy slidedemo
//
// Resolve optional bibtex citations:
settings.twice=true;
texpreamble("\bibliographystyle{alpha}");

import slide;

usersetting();

// Optional background color:
// import x11colors;
// fill(background,box((-1,-1),(1,1)),Azure);

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
label(pic2,"$\pi$",(0,0),fontsize(500));
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

bibliography("refs");

// Slide demo.
// Command-line options to enable stepping and/or reverse video:
// asy [-u stepping=true] [-u reverse=true] slidedemo

orientation=Landscape;

settings.tex="pdflatex";

import slide;

// Optional movie modules:
import pdfanim;     // For portable embedded PDF movies (version 0.53 or later)
import external;    // For portable external movies
import embed;       // For non-portable embedded movies

usersetting();

titlepage("Slides with {\tt Asymptote}: Animations","John C. Bowman",
	  "University of Alberta","\today","http://asymptote.sf.net");

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
// Generated needed files if they don't already exist.
asy("mpg","animations/wheel");
display(embed.embed("animations/wheel.mpg",
		   "poster,text=wheel.mpg,label=wheel.mpg",
		   20cm,5.6cm));
display(embed.link("animations/wheel.mpg","Play","play"));

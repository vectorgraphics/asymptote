// Slide demo.
// Command-line options to enable stepping and/or reverse video:
// asy [-u stepping=true] [-u reverse=true] slidedemo

orientation=Landscape;

settings.tex="pdflatex";

import slide;

// Optional movie modules:
import animate;     // For portable embedded PDF movies
access external;    // For portable external movies
access embed;       // For non-portable embedded movies

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
display(a.pdf("autoplay,loop,controls",multipage=false));
display(b.pdf("controls",multipage=false));

// Generated needed files if they don't already exist.
asy("mpg","wheel");

title("External Movie (portable)");
display(external.embed("wheel.mpg",
                       "poster,text=wheel.mpg,label=wheel.mpg",20cm,5.6cm));
display(external.link("wheel.mpg","Play","play"));

title("Embedded Movie (not portable)");
display(embed.embed("wheel.mpg",
                    "poster,text=wheel.mpg,label=wheel.mpg",
                    20cm,5.6cm));
display(embed.link("wheel.mpg","Play","play"));

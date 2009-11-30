// An embedded movie;
//
// See http://www.tug.org/tex-archive/macros/latex/contrib/movie15/README
// for documentation of the options.

import embed;       // Add embedded movie
//import external;  // Add external movie (use this form under Linux).

// Generated needed mpeg file if it doesn't already exist.
asy("mpg","wheel");

// Produce a pdf file.
settings.outformat="pdf";

settings.twice=true;

// An embedded movie:
label(embed("wheel.mpg","poster,text=wheel.mpg,label=wheel.mpg",20cm,5.6cm),
      (0,0),N);

// An optional button:
label(link("wheel.mpg","Play","play"),(0,0),S);



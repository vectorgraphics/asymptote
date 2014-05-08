// An embedded movie;
//
// See http://mirror.ctan.org/macros/latex/contrib/media9/doc/media9.pdf
// for documentation of the options.

import embed;       // Add embedded movie
//import external;  // Add external movie (use this form under Linux).

// Generated needed mp4 file if it doesn't already exist.
asy("mp4","wheel");

// Produce a pdf file.
settings.outformat="pdf";

settings.twice=true;

// An embedded movie:
label(embed("wheel.mp4",20cm,5.6cm),(0,0),N);
label(link("wheel.mp4"),(0,0),S);

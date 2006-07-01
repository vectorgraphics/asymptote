// An embedded movie;
//
// See http://www.tug.org/tex-archive/macros/latex/contrib/movie15/README
// for documentation of the options.

import embed;

// Generated needed mpeg file if it doesn't already exist.
asy("mpg","wheel");

// Produce a pdf file.
settings.outformat="pdf";

// Run LaTeX twice to resolve references.
settings.twice=true;

label(embed("wheel.mpg","poster,text=wheel.mpg,label=wheel",20cm,5.6cm),
      (0,0),N);

// Optional buttons can be added like this.
label(hyperlink("wheel","Play","play"),(0,0),S);


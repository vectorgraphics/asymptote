// An embedded movie.
//
// The mpeg file for this example can be generated like this:
// asy -f mpg wheel
//
// See http://www.tug.org/tex-archive/macros/latex/contrib/movie15/README
// for documentation of the options.

import embed;
access settings;
settings.outformat="pdf";

label(embed("wheel.mpg","poster,text=wheel.mpg,label=wheel",20cm,5.6cm),
      (0,0),N);

// Optional buttons can be added like this.
label(hyperlink("wheel","Play","play"),(0,0),S);

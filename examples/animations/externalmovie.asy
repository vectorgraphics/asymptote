// Embed a movie to be run in an external window.

import external;

// External movies require the pdflatex engine.
settings.tex="pdflatex";

// Generated needed mpeg file if it doesn't already exist.
asy("mpg","wheel");

// Produce a pdf file.
settings.outformat="pdf";

// External movie: viewable even with the Linux version of acroread.
label(embed("wheel.mpg"),(0,0),N);

label(link("wheel.mpg","Play"),(0,0),S);

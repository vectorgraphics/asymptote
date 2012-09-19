// Embed a movie to be run in an external window.

import external;

// External movies require the pdflatex engine.
settings.tex="pdflatex";

// Generated needed mpeg file if it doesn't already exist.
asy("mp4","wheel");

// Produce a pdf file.
settings.outformat="pdf";

// External movie: viewable even with the Linux version of acroread.
label(embed("wheel.mp4"),(0,0),N);
label(link("wheel.mp4"),(0,0),S);

size(200);

settings.tex="pdflatex";

// PostScript Calculator routine to convert from [0,1]x[0,1] to RG:
string redgreen="0";

// PostScript Calculator routine to convert from [0,1]x[0,1] to HS to RGB:
// (http://www.texample.net/tikz/examples/hsv-shading):
string hsv="0.5 sub exch 0.5 sub exch
2 copy 2 copy 0 eq exch 0 eq and { pop pop 0.0 } {atan 360.0 div}
ifelse dup 360 eq { pop 0.0 }{} ifelse 3 1 roll dup mul exch dup mul add
sqrt 2.5 mul 0.25 sub 1 1 index 1.0
eq { 3 1 roll pop pop dup dup } { 3 -1 roll 6.0 mul dup 4 1 roll floor dup
5 1 roll 3 index sub neg 1.0 3 index sub 2 index mul 6 1 roll dup 3 index
mul neg 1.0 add 2 index mul 7 1 roll neg 1.0 add 2 index mul neg 1.0 add 1
index mul 7 2 roll pop pop dup 0 eq { pop exch pop } { dup 1 eq { pop exch
4 1 roll exch pop } { dup 2 eq { pop 4 1 roll pop } { dup 3 eq { pop exch 4
2 roll pop } { dup 4 eq { pop exch pop 3 -1 roll } { pop 3 1 roll exch pop
} ifelse } ifelse } ifelse } ifelse } ifelse } ifelse cvr 3 1 roll cvr 3 1
roll cvr 3 1 roll";

path p=unitcircle;
functionshade(p,rgb(zerowinding),redgreen);
layer();
draw(p);

path g=shift(2*dir(-45))*p;
functionshade(g,rgb(zerowinding),hsv);
layer();
draw(g);

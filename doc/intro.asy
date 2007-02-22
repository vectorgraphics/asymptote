// Introduction to Asymptote

orientation=Landscape;

import slide;
autorotation=false;
usersetting();

bibliographystyle("alpha");

itempen=fontsize(22pt);

titlepage("Asymptote: The Vector Graphics Language","John C. Bowman",
	  "University of Alberta","\today","http://asymptote.sf.net");

title("Cartesian Coordinates");
asycode("../doc/diagonal");
item("units are {\tt PostScript} {\it big points\/} (1 {\tt bp} =
1/72 {\tt inch})");
item("{\tt --} means join the points with a linear segment to create
a {\it path}");

item("cyclic path:");

asycode("square");


title("Scaling to a Given Size");

item("{\tt PostScript} units are often inconvenient.");

item("Instead, scale user coordinates to a specified final size:");

code("size(101,101);
draw((0,0)--(1,0)--(1,1)--(0,1)--cycle);");
asy("square.asy");
figure("square."+nativeformat());

item("One can also specify the size in {\tt cm}:");

asycode("bigsquare");


title("Labels");

item("Adding and aligning \LaTeX\ labels is easy:");

asycode("labelsquare","height=6cm");


title("Bezier Splines");

item("Using {\tt ..} instead of {\tt --} specifies a {\it Bezier cubic spline\/} \cite{Hobby86,Knuth86b}:");

code("draw(z0..controls c0 and c1 .. z1,blue+dashed);");
asy("beziercurve.asy");
figure("beziercurve."+nativeformat());

equation("(1-t)^3 z_0+3t(1-t)^2 c_0+3t^2(1-t) c_1+t^3 z_1, \qquad t\in [0,1].");


title("Rendering: Midpoint Property");

item("Third-order midpoint $m_5$ is the midpoint of the Bezier curve formed by the quadruple ($z_0$, $c_0$, $c_1$, $z_1$).");

figure("bezier2");

item("Recursively construct the desired curve, using the newly extracted third-order midpoint as an endpoint and the respective second- and first-order midpoints as control points.");


title("{\tt C++}-like Programming Syntax");

code("// Declaration: Declare x to be real:
real x;

// Assignment: Assign x the value 1.
x=1.0;

// Conditional: Test if x equals 1 or not.
if(x == 1.0) {
  write(\"x equals 1.0\");
} else {
  write(\"x is not equal to 1.0\");
}

// Loop: iterate 10 times
for(int i=0; i < 10; ++i) {
  write(i);
}");


title("Textbook Graph");
asy(nativeformat(),"exp.asy");
filecode("exp.asy");
label(graphic("exp."+nativeformat(),"height=10cm"),(0.5,0),
      Fill(figureborder,figuremattpen));


title("Scientific Graph");
asycode("lineargraph","height=13cm",newslide=true);


title("Data Graph");
asycode("datagraph","height=13cm",newslide=true);


title("Imported Data Graph");
asycode("filegraph","height=15cm",newslide=true);


title("Logarithmic Graph");
asycode("loggraph","height=15cm",newslide=true);


title("Secondary Axis");
asyfigure("secondaryaxis","height=15cm");


title("Images");
asyfigure("imagecontour","height=17cm");


title("Multiple graphs");
asyfigure("diatom","height=15cm");

bibliography("refs");

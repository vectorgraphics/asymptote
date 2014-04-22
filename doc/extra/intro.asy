orientation=Landscape;

settings.tex="pdflatex";

import slide;
import three;
import animate;

bool long=true;

usepackage("mflogo");

usersetting();

viewportsize=pagewidth-2pagemargin;

// To generate bibliographic references:
// asy -k intro
// bibtex intro_ 
// asy -k intro
bibliographystyle("alpha");

itempen=fontsize(22pt);
defaultpen(itempen);
viewportmargin=(2,2);

titlepage(long ? "Asymptote: The Vector Graphics Language" :
          "Interactive TeX-Aware 3D Vector Graphics",
          "John Bowman and Andy Hammerlindl",
"Department of Mathematical and Statistical Sciences\\
          University of Alberta\\
%and Instituto Nacional de Matem\'atica Pura e Aplicada (IMPA)
\medskip\Green{Collaborators: Orest Shardt, Michail Vidiassov}",
"June 30, 2010",
"http://asymptote.sf.net/intro.pdf");

title("History");
item("1979: \TeX\ and \MF\ (Knuth)");
item("1986: 2D B\'ezier control point selection (Hobby)");
item("1989: MetaPost (Hobby)");
item("2004: Asymptote");
subitem("2004: initial public release (Hammerlindl, Bowman, \& Prince)");
subitem("2005: 3D B\'ezier control point selection (Bowman)");
subitem("2008: 3D interactive \TeX\ within PDF files (Shardt \& Bowman)");
subitem("2009: 3D billboard labels that always face camera (Bowman)");
subitem("2010: 3D PDF enhancements (Vidiassov \& Bowman)");

title("Statistics (as of June, 2010)");
item("Runs under Linux/UNIX, Mac OS X, Microsoft Windows.");
item("4000 downloads/month from primary\hfill\\
 {\tt asymptote.sourceforge.net} site alone.");
item("80\ 000 lines of low-level C++ code.");
item("36\ 000 lines of high-level Asymptote code.");

if(long) {
title("Vector Graphics");
item("Raster graphics assign colors to a grid of pixels.");
figure("pixel.pdf");
item("Vector graphics are graphics which still maintain their look when
    inspected at arbitrarily small scales.");
asyfigure(asywrite("
picture pic;

path zoombox(real h) {
  return box((-h,-h/2),(min(10,h),min(10,h)/2));
}

frame zoom(real h, real next=0) {
  frame f;
  draw(f, (0,-100){W}..{E}(0,0), Arrow);
  clip(f, zoombox(h));
  if(next > 0)
    draw(f, zoombox(next));

  return scale(100/h)*f;
}

add(zoom(100), (0,0));
add(zoom(10), (200,0));
add(zoom(1), (400,0));
"));
}

title("Cartesian Coordinates");

item("Asymptote's graphical capabilities are based on four primitive
    commands: {\tt draw}, {\tt label}, {\tt fill}, {\tt clip} \cite{Bowman08}");

asyfilecode("diagonal");
item("units are {\tt PostScript} {\it big points\/} (1 {\tt bp} =
1/72 {\tt inch})");
item("{\tt --} means join the points with a linear segment to create
a {\it path}");

item("{\it cyclic\/} path:");

asycode("
draw((0,0)--(100,0)--(100,100)--(0,100)--cycle);
");

title("Scaling to a Given Size");

item("{\tt PostScript} units are often inconvenient.");

item("Instead, scale user coordinates to a specified final size:");

asyfilecode("square");

item("One can also specify the size in {\tt cm}:");

asycode("
size(3cm,3cm);
draw(unitsquare);
");

title("Labels");

item("Adding and aligning \LaTeX\ labels is easy:");

asycode(preamble="defaultpen(fontsize("+string(fontsize(itempen))+"));",
"size(6cm);
draw(unitsquare);
label(\"$A$\",(0,0),SW);
label(\"$B$\",(1,0),SE);
label(\"$C$\",(1,1),NE);
label(\"$D$\",(0,1),NW);
");

title("2D B\'ezier Splines");

item("Using {\tt ..} instead of {\tt --} specifies a {\it B\'ezier cubic
spline}:");

code("
draw(z0 .. controls c0 and c1 .. z1,blue);
");
asyfigure(asywrite("defaultpen(fontsize("+string(fontsize(itempen))+"));
size(0,7cm);
pair z0=(0,0);
pair c0=(1,1);
pair c1=(2,1);
pair z1=(3,0);
draw(z0..controls c0 and c1 .. z1,blue);
draw(z0--c0--c1--z1,dashed);
dot(\"$z_0$\",z0,W,red);
dot(\"$c_0$\",c0,NW,red);
dot(\"$c_1$\",c1,NE,red);
dot(\"$z_1$\",z1,red);
"));

equation("(1-t)^3 z_0+3t(1-t)^2 c_0+3t^2(1-t) c_1+t^3 z_1, \qquad t\in [0,1].");

title("Smooth Paths");

item("Asymptote can choose control points for you, using the algorithms of
Hobby and Knuth \cite{Hobby86,Knuth86b}:");

string bean="
pair[] z={(0,0), (0,1), (2,1), (2,0), (1,0)};
";

asycode(preamble="size(130,0);",bean+"
draw(z[0]..z[1]..z[2]..z[3]..z[4]..cycle,
     grey+linewidth(5));
dot(z,linewidth(7));
");

item("First, linear equations involving the curvature are solved to find the
    direction through each knot.  Then, control points along those directions
    are chosen:");

asyfigure(asywrite(preamble="size(130,0);",bean+"
path p=z[0]..z[1]..z[2]..z[3]..z[4]..cycle;

dot(z);
draw(p,lightgrey+linewidth(5));
dot(z);

picture output;
save();
for(int i=0; i<length(p); ++i) {
  pair z=point(p,i), dir=dir(p,i);
  draw((z-0.3dir)--(z+0.3dir), Arrow);
}
add(output, currentpicture.fit(), (-0.5inch, 0), W);
restore();

save();
guide g;
for(int i=0; i<length(p); ++i) {
  dot(precontrol(p,i));
  dot(postcontrol(p,i));
  g=g--precontrol(p,i)--point(p,i)--postcontrol(p,i);
}
draw(g--cycle,dashed);
add(output, currentpicture.fit(), (+0.5inch, 0), E);
restore();

shipout(output);
"));

title("Filling");
item("The {\tt fill} primitive to fill the inside of a path:");
asycode(preamble="size(0,200);","
path star;
for(int i=0; i < 5; ++i)
  star=star--dir(90+144i);
star=star--cycle;

fill(star,orange+zerowinding);
draw(star,linewidth(3));

fill(shift(2,0)*star,blue+evenodd);
draw(shift(2,0)*star,linewidth(3));
");

title("Filling");
item("Use a list of paths to fill a region with holes:");
asycode(preamble="size(0,300);","
path[] p={scale(2)*unitcircle, reverse(unitcircle)};
fill(p,green+zerowinding);
");

title("Clipping");
item("Pictures can be clipped to a path:");
asycode(preamble="
size(0,200);
guide star;
for(int i=0; i<5; ++i)
  star=star--dir(90+144i);
star=star--cycle;","
fill(star,orange+zerowinding);
clip(scale(0.7)*unitcircle);
draw(scale(0.7)*unitcircle);
");

title("Affine Transforms");

item("Affine transformations: shifts, rotations, reflections, and scalings
        can be applied to pairs, paths, pens, strings, and even whole pictures:");

code("
fill(P,blue);
fill(shift(2,0)*reflect((0,0),(0,1))*P, red);
fill(shift(4,0)*rotate(30)*P, yellow);
fill(shift(6,0)*yscale(0.7)*xscale(2)*P, green);
");
asyfigure(asywrite("
size(500,0);
real bw=0.15;
real sw=0.2;
real r=0.15;

path outside=(0,0)--(0,1)--
    (bw+sw,1)..(bw+sw+r+bw,1-(r+bw))..(bw+sw,1-2(r+bw))--
    (bw,1-2(r+bw))--(bw,0)--cycle;
path inside=(bw,1-bw-2r)--(bw,1-bw)--
    (bw+sw,1-bw)..(bw+sw+r,1-bw-r)..(bw+sw,1-bw-2r)--cycle;
//fill(new path[] {outside, reverse(inside)},yellow);

path[] P={outside, reverse(inside)};

fill(P,blue);
fill(shift(2,0)*reflect((0,0),(0,1))*P, red);
fill(shift(4,0)*rotate(30)*P, yellow);
fill(shift(6,0)*yscale(0.7)*xscale(2)*P, green);
"));

if(long) {
title("C++/Java-like Programming Syntax");

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
}

title("Modules");

item("There are modules for Feynman diagrams,");
asyfigure("eetomumu","height=6cm");
remark("data structures,");
asyfigure(asywrite("
import binarytree;

binarytree bt=binarytree(1,2,4,nil,5,nil,nil,0,nil,nil,3,6,nil,nil,7);
draw(bt);
"),"height=6cm");
newslide();
remark("algebraic knot theory:");
asyfigure("knots");
equations("\Phi\Phi(x_1,x_2,x_3,x_4,x_5)
    =   &\rho_{4b}(x_1+x_4,x_2,x_3,x_5) + \rho_{4b}(x_1,x_2,x_3,x_4) \\
      + &\rho_{4a}(x_1,x_2+x_3,x_4,x_5) - \rho_{4b}(x_1,x_2,x_3,x_4+x_5) \\
      - &\rho_{4a}(x_1+x_2,x_3,x_4,x_5) - \rho_{4a}(x_1,x_2,x_4,x_5).");

if(long) {
title("Textbook Graph");
asy(nativeformat(),"exp");
filecode("exp.asy");
label(graphic("exp."+nativeformat(),"height=10cm"),(0.5,0),
      Fill(figureborder,figuremattpen));

title("Scientific Graph");
asyfilecode("lineargraph","height=13cm",newslide=true);

title("Data Graph");
asyfilecode("datagraph","height=13cm",newslide=true);

title("Imported Data Graph");
asyfilecode("filegraph","height=15cm",newslide=true);

title("Logarithmic Graph");
asyfilecode("loggraph","height=15cm",newslide=true);
title("Secondary Axis");
} else
title("Scientific Graph");

asyfigure("secondaryaxis","height=15cm");

title("Images and Contours");
asyfigure("imagecontour","height=17cm");

title("Multiple Graphs");
asyfigure("diatom","height=17cm");

title("Hobby's 2D Direction Algorithm");
item("A tridiagonal system of linear equations is solved to determine any unspecified directions $\phi_k$ and $\theta_k$ through each knot $z_k$:");

equation("\frac{\theta_{k-1}-2\phi_k}{\ell_k}=
\frac{\phi_{k+1}-2\theta_k}{\ell_{k+1}}.");

asyfigure("Hobbydir","height=9cm");

item("The resulting shape may be adjusted by modifying optional {\it tension\/} parameters and {\it curl\/} boundary conditions.");

title("Hobby's 2D Control Point Algorithm");
item("Having prescribed outgoing and incoming path directions $e^{i\theta}$
at node~$z_0$ and $e^{i\phi}$ at node $z_1$ relative to the
vector $z_1-z_0$, the control points are determined as:");  

equations("u&=&z_0+e^{i\theta}(z_1-z_0)f(\theta,-\phi),\nonumber\\
v&=&z_1-e^{i\phi}(z_1-z_0)f(-\phi,\theta),");

remark("where the relative distance function $f(\theta,\phi)$ is given by Hobby [1986].");

asyfigure("Hobbycontrol","height=9cm");

if(long) {
title("B\'ezier Curves in 3D");

item("Apply an affine transformation");

equation("x'_i=A_{ij} x_j+C_i");

remark("to a B\'ezier curve:");

equation("\displaystyle x(t)=\sum_{k=0}^3 B_k(t) P_k, \qquad t\in [0,1].");

item("The resulting curve is also a B\'ezier curve:");

skip(-2);

equations("x'_i(t)&=&\sum_{k=0}^3 B_k(t) A_{ij}(P_k)_j+C_i\nonumber\\
&=&\sum_{k=0}^3 B_k(t) P'_k,");

skip(-2);

remark("where $P'_k$ is the transformed $k^{\rm th}$ control point, noting
$\displaystyle\sum_{k=0}^3 B_k(t)=1.$");
}

title("3D Generalization of Direction Algorithm");

item("Must reduce to 2D algorithm in planar case.");

item("Determine directions by applying Hobby's algorithm in the plane containing $z_{k-1}$, $z_k$, $z_{k+1}$.");

// Reformulate Hobby's equations in terms of the angle $\psi_k=$
item("The only ambiguity that can arise is the overall sign of the angles, which relates to viewing each 2D plane from opposing normal directions."); 

item("A reference vector based on the mean unit normal of successive segments can be used to resolve such ambiguities \cite{Bowman07,Bowman09}");

title("3D Control Point Algorithm");

item("Express Hobby's algorithm in terms of the absolute directions $\omega_0$ and~$\omega_1$:");
skip(-1);
equation("u=z_0+\omega_0\left|z_1-z_0\right|f(\theta,-\phi),");
equation("v=z_1-\omega_1\left|z_1-z_0\right|f(-\phi,\theta),");

asyfigure("Hobbycontrol");

remark("interpreting $\theta$ and $\phi$ as the angle between the corresponding path direction vector and $z_1-z_0$.");

item("Here there is an unambiguous reference vector for determining the relative sign of the angles $\phi$ and $\theta$.");

viewportmargin=(2,0.5cm);
//defaultpen(1.0);
title("Interactive 3D Saddle");
item("A unit circle in the $X$--$Y$ plane may be constructed with:
{\tt (1,0,0)..(0,1,0)..(-1,0,0)..(0,-1,0)..cycle}:");
asyinclude("unitcircle3",8cm);
remark("and then distorted into the saddle\\
{\tt (1,0,0)..(0,1,1)..(-1,0,0)..(0,-1,1)..cycle}:");
asyinclude("saddle",8cm);
//defaultpen(0.5);

title("Lifting TeX to 3D");
item("Glyphs are first split into simply connected regions and then decomposed into planar B\'ezier surface patches \cite{Bowman09,Shardt12}:");
asyfigure("../examples/partitionExample");

viewportmargin=(2,1cm);
title("Label Manipulation");
item("They can then be extruded and/or arbitrarily transformed:");
asyinclude("../examples/label3solid");

title("Billboard Labels");
defaultpen(fontsize(36pt));
asyinclude("../examples/billboard",15cm);
defaultpen(itempen);

title("Smooth 3D surfaces");
asyinclude("../examples/sinc",25cm);

title("Curved 3D Arrows");
asyinclude("../examples/arrows3",20cm);

title("Slide Presentations");
item("Asymptote has a module for preparing slides.");
item("It even supports embedded high-resolution PDF movies.");

code('
title("Slide Presentations");
item("Asymptote has a module for preparing slides.");
item("It even supports embedded high-resolution PDF movies.");
');
remark("\quad\ldots");

import graph;

pen p=linewidth(1);
pen dotpen=linewidth(5);

pair wheelpoint(real t) {return (t+cos(t),-sin(t));}

guide wheel(guide g=nullpath, real a, real b, int n)
{
  real width=(b-a)/n;
  for(int i=0; i <= n; ++i) {
    real t=a+width*i;
    g=g--wheelpoint(t);
  }
  return g;
}

real t1=0; 
real t2=t1+2*pi;

picture base;
draw(base,circle((0,0),1),p);
draw(base,wheel(t1,t2,100),p+linetype("0 2"));
yequals(base,Label("$y=-1$",1.0),-1,extend=true,p+linetype("4 4"));
xaxis(base,Label("$x$",align=3SW),0,p);
yaxis(base,"$y$",0,1.3,p);
pair z1=wheelpoint(t1);
pair z2=wheelpoint(t2);
dot(base,z1,dotpen);
dot(base,z2,dotpen);

animation a;

int n=25;
real dt=(t2-t1)/n;
for(int i=0; i <= n; ++i) {
  picture pic;
  size(pic,24cm);
  real t=t1+dt*i;
  add(pic,base);
  draw(pic,circle((t,0),1),p+red);
  dot(pic,wheelpoint(t),dotpen);
  a.add(pic);
}

display(a.pdf(delay=150,"controls"));

title("Automatic Sizing");
item("Figures can be specified in user coordinates, then
    automatically scaled to the desired final size.");
asyfigure(asywrite("
import graph;

size(0,100);

frame cardsize(real w=0, real h=0, bool keepAspect=Aspect) {
  picture pic;
  pic.size(w,h,keepAspect);

  real f(real t) {return 1+cos(t);}

  guide g=polargraph(f,0,2pi,operator ..)--cycle;
  filldraw(pic,g,pink);

  xaxis(pic,\"$x$\",above=true);
  yaxis(pic,\"$y$\",above=true);

  dot(pic,\"$(a,0)$\",(1,0),N);
  dot(pic,\"$(2a,0)$\",(2,0),N+E);

  frame f=pic.fit();
  label(f,\"{\tt size(\"+string(w)+\",\"+string(h)+\");}\",point(f,S),align=S);
  return f;
}

add(cardsize(0,50), (0,0));
add(cardsize(0,100), (230,0));
add(cardsize(0,200), (540,0));
"));

title("Deferred Drawing");
item("We can't draw a graphical object until we know the scaling
    factors for the user coordinates.");
item("Instead, store a function that, given the scaling information, draws
    the scaled object.");
code("
void draw(picture pic=currentpicture, path g, pen p=currentpen) {
  pic.add(new void(frame f, transform t) {
      draw(f,t*g,p);
    });
  pic.addPoint(min(g),min(p));
  pic.addPoint(max(g),max(p));
}
");

title("Coordinates");
item("Store bounding box information as the sum of user and true-size
    coordinates:");
asyfigure(asywrite("
size(0,150);

path q=(0,0){dir(70)}..{dir(70)}(100,50);
pen p=rotate(30)*yscale(0.7)*(lightblue+linewidth(20));
draw(q,p);
draw((90,10),p);

currentpicture.add(new void(frame f, transform t) {
    draw(f,box(min(t*q)+min(p),max(t*q)+max(p)), dashed);
    });

draw(box(min(q),max(q)));

frame f;
draw(f,box(min(p),max(p)));

add(f,min(q));
add(f,max(q));

draw(q);
"));

code("pic.addPoint(min(g),min(p));
pic.addPoint(max(g),max(p));");
item("Filling ignores the pen width:");
code("pic.addPoint(min(g),(0,0));
pic.addPoint(max(g),(0,0));");
item("Communicate with \LaTeX\ {\it via\/} a pipe to determine label sizes:");

asyfigure(asywrite("
size(0,100);

pen p=fontsize(30pt);
frame f;
label(f, \"$E=mc^2$\", p);
draw(f, box(min(f),max(f)));
shipout(f);
"));

title("Sizing");

item("When scaling the final figure to a given size $S$, we first need to
    determine a scaling factor $a>0$ and a shift $b$ so that all of the
    coordinates when transformed will lie in the interval $[0,S]$.");

item("That is, if $u$ and $t$ are the user and truesize components:");
equation("0\le au+t+b \le S.");

item("Maximize the variable $a$ subject to a number of inequalities.");

item("Use the simplex method to solve the resulting linear programming problem.");

if(long) {
title("Sizing");
item("Every addition of a coordinate $(t,u)$ adds two restrictions");
equation("au+t+b\ge 0,");
equation("au+t+b\le S,");
remark("and each drawing component adds two coordinates.");
item("A figure could easily produce thousands of restrictions, making the
    simplex method impractical.");

item("Most of these restrictions are redundant, however.  For instance, with
    concentric circles, only the largest circle needs to be accounted for.");
asyfigure(asywrite("
import palette;
size(160,0);
pen[] p=Rainbow(NColors=11);
for(int i=1; i<10; ++i) {
  draw(scale(i)*unitcircle, p[i]+linewidth(2));
}
"));

title("Redundant Restrictions");
item("In general, if $u\le u'$ and $t\le t'$ then");
equation("au+t+b\le au'+t'+b");
remark("for all choices of $a>0$ and $b$, so");
equation("0\le au+t+b\le au'+t'+b\le S.");
item("This defines a partial ordering on coordinates.  When sizing a picture,
    the program first computes which coordinates are maximal (or minimal) and
    only sends effective constraints to the simplex algorithm.");
item("In practice, the linear programming problem will have less than a dozen
    restraints.");
item("All picture sizing is implemented in Asymptote code.");
}

title("Infinite Lines");
item("Deferred drawing allows us to draw infinite lines.");
code("drawline(P, Q);");

asyfigure("elliptic","height=12cm");

title("Helpful Math Notation");

item("Integer division returns a {\tt real}.  Use {\tt quotient} for an integer
    result:");
code("3/4 == 0.75         quotient(3,4) == 0");

item("Caret for real and integer exponentiation:");
code("2^3    2.7^3    2.7^3.2");

item("Many expressions can be implicitly scaled by a numeric constant:");
code("2pi    10cm    2x^2    3sin(x)    2(a+b)");

item("Pairs are complex numbers:");
code("(0,1)*(0,1) == (-1,0)");

title("Function Calls");

item("Functions can take default arguments in any position.  Arguments are
    matched to the first possible location:");
string unitsize="unitsize(0.65cm);";
string preamble="void drawEllipse(real xsize=1, real ysize=xsize, pen p=blue) {
  draw(xscale(xsize)*yscale(ysize)*unitcircle, p);
}
";

asycode(preamble=unitsize,preamble+"
drawEllipse(2);
drawEllipse(red);
");

item("Arguments can be given by name:");
asycode(preamble=unitsize+preamble,"
drawEllipse(xsize=2, ysize=1);
drawEllipse(ysize=2, xsize=3, green);
");

if(long) {
title("Rest Arguments");
item("Rest arguments allow one to write a function that takes an arbitrary
    number of arguments:");
code("
int sum(... int[] nums) {
  int total=0; 
  for(int i=0; i < nums.length; ++i)
    total += nums[i];
  return total;
}

sum(1,2,3,4);                       // returns 10
sum();                              // returns 0
sum(1,2,3 ... new int[] {4,5,6});   // returns 21

int subtract(int start ... int[] subs) {
  return start - sum(... subs);
}
");
}

title("High-Order Functions");

item("Functions are first-class values.  They can be passed to other
    functions:");
code("import graph;
real f(real x) {
    return x*sin(10x);
}
draw(graph(f,-3,3,300),red);");
asyfigure(asywrite("
import graph;
size(300,0);
real f(real x) {
    return x*sin(10x);
}
draw(graph(f,-3,3,300),red);
"));

if(long) {
title("Higher-Order Functions");
item("Functions can return functions:");
equation("f_n(x)=n\sin\left(\frac{x}{n}\right).");
skip();
string preamble="
import graph;
size(300,0);
";
string graphfunc2="
typedef real func(real);
func f(int n) {
  real fn(real x) {
    return n*sin(x/n);
  }
  return fn;
}

func f1=f(1);
real y=f1(pi);

for(int i=1; i<=5; ++i)
  draw(graph(f(i),-10,10),red);
";
code(graphfunc2);
string name=asywrite(graphfunc2,preamble=preamble);
asy(nativeformat(),name+".asy");
label(graphic(name+"."+nativeformat()),(0.5,0),
      Fill(figureborder,figuremattpen));

title("Anonymous Functions");

item("Create new functions with {\tt new}:");
code("
path p=graph(new real (real x) { return x*sin(10x); },-3,3,red);

func f(int n) {
  return new real (real x) { return n*sin(x/n); };
}");

item("Function definitions are just syntactic sugar for assigning function
objects to variables.");
code("
real square(real x) {
  return x^2;
}
");

remark("is equivalent to");
code("
real square(real x);
square=new real (real x) {
  return x^2;
};
");

title("Structures");

item("As in other languages, structures group together data.");
code("
struct Person {
  string firstname, lastname;
  int age;
}
Person bob=new Person;
bob.firstname=\"Bob\";
bob.lastname=\"Chesterton\";
bob.age=24;
");

item("Any code in the structure body will be executed every time a new structure
    is allocated...");
code("
struct Person {
  write(\"Making a person.\");
  string firstname, lastname;
  int age=18;
}
Person eve=new Person;   // Writes \"Making a person.\"
write(eve.age);          // Writes 18.
");

title("Modules");

item("Function and structure definitions can be grouped into modules:");
code("
// powers.asy
real square(real x) { return x^2; }
real cube(real x) { return x^3; }
");
remark("and imported:");
code("
import powers;
real eight=cube(2.0);
draw(graph(powers.square, -1, 1));
");
}

title("Object-Oriented Programming");
item("Functions are defined for each instance of a structure.");
code("
struct Quadratic {
  real a,b,c;
  real discriminant() {
    return b^2-4*a*c;
  }
  real eval(real x) {
    return a*x^2 + b*x + c;
  }
}
");

item("This allows us to construct ``methods'' which are just normal functions
    declared in the environment of a particular object:");
code("
Quadratic poly=new Quadratic;
poly.a=-1; poly.b=1; poly.c=2;

real f(real x)=poly.eval;
real y=f(2);
draw(graph(poly.eval, -5, 5));
");

title("Specialization");

item("Can create specialized objects just by redefining methods:");
code("
struct Shape {
    void draw();
    real area();
}

Shape rectangle(real w, real h) {
  Shape s=new Shape;
  s.draw = new void () {
                   fill((0,0)--(w,0)--(w,h)--(0,h)--cycle); };
  s.area = new real () { return w*h; };
  return s;
}

Shape circle(real radius) {
  Shape s=new Shape;
  s.draw = new void () { fill(scale(radius)*unitcircle); };
  s.area = new real () { return pi*radius^2; }
  return s;
}
");

title("Overloading");
item("Consider the code:");
code("
int x1=2;
int x2() {
  return 7;
}
int x3(int y) {
  return 2y;
}

write(x1+x2());  // Writes 9.
write(x3(x1)+x2());  // Writes 11.
");

title("Overloading");
item("{\tt x1}, {\tt x2}, and {\tt x3} are never used in the same context, so
    they can all be renamed {\tt x} without ambiguity:");
code("
int x=2;
int x() {
  return 7;
}
int x(int y) {
  return 2y;
}

write(x+x());  // Writes 9.
write(x(x)+x());  // Writes 11.
");

item("Function definitions are just variable definitions, but variables are
    distinguished by their signatures to allow overloading.");

title("Operators");
item("Operators are just syntactic sugar for functions, and can be addressed or
    defined as functions with the {\tt operator} keyword.");
code("
int add(int x, int y)=operator +;
write(add(2,3));  // Writes 5.

// Don't try this at home.
int operator +(int x, int y) {
  return add(2x,y);
}
write(2+3);  // Writes 7.
");
item("This allows operators to be defined for new types.");

title("Operators");
item("Operators for constructing paths are also functions:");
code("a.. controls b and c .. d--e");
remark("is equivalent to");
code(
     "operator --(operator ..(a, operator controls(b,c), d), e)");
item("This allowed us to redefine all of the path operators for 3D paths.");

title("Summary");

item("Asymptote:");
subitem("uses IEEE floating point numerics;");
subitem("uses C++/Java-like syntax;");
subitem("supports deferred drawing for automatic picture sizing;");
subitem("supports Grayscale, RGB, CMYK, and HSV colour spaces;");
subitem("supports PostScript shading, pattern fills, and function shading;");
subitem("can fill nonsimply connected regions;");
subitem("generalizes MetaPost path construction algorithms to 3D;");
subitem("lifts \TeX\ to 3D;");
subitem("supports 3D billboard labels and PDF grouping.");

bibliography("../examples/refs");

viewportmargin=(2,2);
viewportsize=0;
defaultpen(0.5);
title("\mbox{Asymptote: 2D \& 3D Vector Graphics Language}");
asyinclude("../examples/logo3");
skip();
center("\tt http://asymptote.sf.net");
center("(freely available under the LGPL license)");

//  LocalWords:  pdflatex mflogo viewportsize pagewidth pagemargin goysr bibtex
//  LocalWords:  itempen defaultrender medskip Orest Shardt Vidiassov MF ezier
//  LocalWords:  Hammerlindl MetaPost PDF hfill LGPL pdf asywrite zoombox LaTeX
//  LocalWords:  asyfilecode PostScript asycode unitsquare beziercurve grey bw
//  LocalWords:  lightgrey zerowinding evenodd sw unitsize drawEllipse nums fn
//  LocalWords:  frac graphfunc func nativeformat figureborder figuremattpen bt
//  LocalWords:  firstname lastname eval eetomumu binarytree filecode datagraph
//  LocalWords:  lineargraph filegraph loggraph secondaryaxis imagecontour ij
//  LocalWords:  tridiagonal Hobbydir nonumber Hobbycontrol th viewportmargin
//  LocalWords:  asyinclude dotpen wheelpoint yequals xaxis yaxis cardsize mc
//  LocalWords:  polargraph filldraw addPoint lightblue truesize le au NColors
//  LocalWords:  drawline unityroot mult oct intang IEEE numerics HSV colour
//  LocalWords:  nonsimply

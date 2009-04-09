size(200);
import labelpath3;

path3 g=(1,0,0)..(0,1,1)..(-1,0,0)..(0,-1,1)..cycle;
path3 g2=shift(-Z)*reverse(unitcircle3);

string txt1="\hbox{This is a test of \emph{curved} 3D labels in
\textbf{Asymptote} (implemented with {\tt texpath}).}";

string txt2="This is a test of curved labels in Asymptote\\(implemented
without the {\tt PSTricks pstextpath} macro)."; 

draw(surface(g),paleblue+opacity(0.5));
draw(labelpath(txt1,subpath(g,0,reltime(g,0.95)),angle=-90),orange);

draw(g2,1bp+red);
draw(labelpath(txt2,subpath(g2,0,3.9),angle=180,optional=rotate(-70,X)*Z));

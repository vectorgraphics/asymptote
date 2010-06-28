import fontsize;
import three;

settings.autobillboard=false;
settings.embed=false;
currentprojection=orthographic(Z);

defaultpen(fontsize(100pt));

dot(O);

label("acg",O,align=N,basealign);
label("ace",O,align=N,red);
label("acg",O,align=S,basealign);
label("ace",O,align=S,red);
label("acg",O,align=E,basealign);
label("ace",O,align=E,red);
label("acg",O,align=W,basealign);
label("ace",O,align=W,red);

picture pic;
dot(pic,(labelmargin(),0,0),blue);
dot(pic,(-labelmargin(),0,0),blue);
dot(pic,(0,labelmargin(),0),blue);
dot(pic,(0,-labelmargin(),0),blue);
add(pic,O);

dot((0,0));

label("acg",(0,0),align=N,basealign);
label("ace",(0,0),align=N,red);
label("acg",(0,0),align=S,basealign);
label("ace",(0,0),align=S,red);
label("acg",(0,0),align=E,basealign);
label("ace",(0,0),align=E,red);
label("acg",(0,0),align=W,basealign);
label("ace",(0,0),align=W,red);

picture pic;
dot(pic,(labelmargin(),0),blue);
dot(pic,(-labelmargin(),0),blue);
dot(pic,(0,labelmargin()),blue);
dot(pic,(0,-labelmargin()),blue);
add(pic,(0,0));

import fontsize; 
import three; 
currentprojection=orthographic(Z); 
 
defaultpen(fontsize(100pt)); 
 
dot((0,0,0)); 
 
label("acg",(0,0,0),align=N,basealign); 
label("ace",(0,0,0),align=N,red); 
label("acg",(0,0,0),align=S,basealign); 
label("ace",(0,0,0),align=S,red); 
label("acg",(0,0,0),align=E,basealign); 
label("ace",(0,0,0),align=E,red); 
label("acg",(0,0,0),align=W,basealign); 
label("ace",(0,0,0),align=W,red); 
 
picture pic; 
dot(pic,(labelmargin(),0,0),blue); 
dot(pic,(-labelmargin(),0,0),blue); 
dot(pic,(0,labelmargin(),0),blue); 
dot(pic,(0,-labelmargin(),0),blue); 
add(pic,(0,0,0)); 
 
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

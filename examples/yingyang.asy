guide center = (0,1){W}..tension 0.8..(0,0){(1,-.5)}..tension 0.8..{W}(0,-1); 

draw((0,1)..(-1,0)..(0,-1));
picture canvas=new picture; 
filldraw(canvas,center{E}..{N}(1,0)..{W}cycle);
fill(canvas,circle((0,0.5),0.125),white);
add(currentpicture,canvas);
fill(circle((0,-0.5),0.125));

shipout(0,23cm);

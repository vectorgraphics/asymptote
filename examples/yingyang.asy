size(0,25cm);
guide center=(0,1){W}..tension 0.8..(0,0){(1,-.5)}..tension 0.8..{W}(0,-1); 

draw((0,1)..(-1,0)..(0,-1));
filldraw(center{E}..{N}(1,0)..{W}cycle);
unfill(circle((0,0.5),0.125));
fill(circle((0,-0.5),0.125));

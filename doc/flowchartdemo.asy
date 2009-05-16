size(0,300);

import flowchart;

block block1=rectangle(Label("Example",magenta),
		       pack(Label("Start:",heavygreen),"",Label("$A:=0$",blue),
			    "$B:=1$"),(-0.5,3),palegreen,paleblue,red);
block block2=diamond(Label("Choice?",blue),(0,2),palegreen,red);
block block3=roundrectangle("Do something",(-1,1));
block block4=bevel("Don't do something",(1,1));
block block5=circle("End",(0,0));

draw(block1);
draw(block2);
draw(block3);
draw(block4);
draw(block5);

add(new void(picture pic, transform t) {
    draw(pic,path(new pair[]{block1.right(t),block2.top(t)},Horizontal),
         Arrow,PenMargin);
    draw(pic,block1.right(t){N}..{W}block2.bottom(t),blue);
    
    draw(pic,Label("Yes",0.5,NW),path(new pair[]{block2.left(t),block3.top(t)},
                                   Horizontal),Arrow,PenMargin);
    draw(pic,Label("No",0.5,NE),path(new pair[]{block2.right(t),block4.top(t)},
                                    Horizontal),Arrow,PenMargin);
    draw(pic,path(new pair[]{block3.bottom(t),block5.left(t)},Vertical),
         Arrow,PenMargin);
    draw(pic,path(new pair[]{block4.bottom(t),block5.right(t)},Vertical),
         Arrow,PenMargin);
  });

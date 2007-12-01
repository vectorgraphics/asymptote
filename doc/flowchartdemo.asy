size(0,300);

import flowchart;

block block1=rectangle("Example",pack("Start:","","$A:=0$","$B:=1$"),(-0.5,3));
block block2=diamond(Label("Choice?",blue),(0,2),palegreen,red);
block block3=roundrectangle("Do something",(-1,1));
block block4=bevel("Don't do something",(1,1));
block block5=circle("End",(0,0));

draw(block1);
draw(block2);
draw(block3);
draw(block4);
draw(block5);

add(new void(frame f, transform t) {
    picture pic;
    draw(pic,path(new pair[]{block1.right(t),block2.top(t)},Horizontal),
         Arrow,PenMargin);
    draw(pic,Label("Yes",0.5),path(new pair[]{block2.left(t),block3.top(t)},
                                   Horizontal),Arrow,PenMargin);
    draw(pic,Label("No",0.5,N),path(new pair[]{block2.right(t),block4.top(t)},
                                    Horizontal),Arrow,PenMargin);
    draw(pic,path(new pair[]{block3.bottom(t),block5.left(t)},Vertical),
         Arrow,PenMargin);
    draw(pic,path(new pair[]{block4.bottom(t),block5.right(t)},Vertical),
         Arrow,PenMargin);

    add(f,pic.fit());
  });


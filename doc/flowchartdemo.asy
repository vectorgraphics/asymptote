import flowchart;

block block1=rectangle("Start",(-50,300));
block block1=rectangle("Example",pack("Start:","","$A:=0$","$B:=1$"),
		       (-50,300));

block block2=diamond("Choice?",(0,200));
block block3=roundrectangle("Do something",(-100,100));
block block4=bevel("Don't do something",(100,100));
block block5=circle("End",(0,0));

draw(block1);
draw(block2);
draw(block3);
draw(block4);
draw(block5);

draw(path(new pair[]{block1.right(),block2.top()},Horizontal),
     Arrow,PenMargin);
draw(Label("Yes",0.5),path(new pair[]{block2.left(),block3.top()},
			   Horizontal),Arrow,PenMargin);
draw(Label("Yes",0.5),path(new pair[]{block2.left(),block3.top()},
			   Horizontal),Arrow,PenMargin);
draw(Label("No",0.5,N),path(new pair[]{block2.right(),block4.top()},
			    Horizontal),Arrow,PenMargin);
draw(path(new pair[]{block3.bottom(),block5.left()},Vertical),
     Arrow,PenMargin);
draw(path(new pair[]{block4.bottom(),block5.right()},Vertical),
     Arrow,PenMargin);

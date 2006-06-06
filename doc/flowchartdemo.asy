import flowchart;

flowblock block1=flowrectangle(picture("Example"),picture("Start"),
			       (-50, 300));
flowblock block2=flowdiamond(picture("Choice?"),(0, 200));
flowblock block3=flowroundrectangle(picture("Do something"),(-100, 100));
flowblock block4=flowbevel(picture("Don't do something"),(100, 100));
flowblock block5=flowcircle(picture("End"),(0, 0));

draw(block1);
draw(block2);
draw(block3);
draw(block4);
draw(block5);

draw(flowpath(new pair[]{block1.right(), block2.top()},
	      new bool[]{true}),Arrow,PenMargin);
draw(Label("Yes",0.5),flowpath(new pair[]{block2.left(), block3.top()},
			       new bool[]{true}),Arrow,PenMargin);
draw(Label("No",0.5,N),flowpath(new pair[]{block2.right(), block4.top()},
				new bool[]{true}),Arrow,PenMargin);
draw(flowpath(new pair[]{block3.bottom(), block5.left()},
	      new bool[]{false}),Arrow,PenMargin);
draw(flowpath(new pair[]{block4.bottom(), block5.right()},
	      new bool[]{false}),Arrow,PenMargin);

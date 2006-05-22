import flowchart;

flowblock block1=flowrectangle(picture("Example"),picture("Start"),
			       (-50, 300));
flowblock block2=flowdiamond(picture("Choice?"),(0, 200));
flowblock block3=flowroundrectangle(picture("Do something"),(-100, 100));
flowblock block4=flowbevel(picture("Don't do something"),(100, 100));
flowblock block5=flowcircle(picture("End"),(0, 0));

path path1=flowpath(new pair[]{block1.right(), block2.top()},
		    new bool[]{true});
path path2=flowpath(new pair[]{block2.left(), block3.top()},
		    new bool[]{true});
path path3=flowpath(new pair[]{block2.right(), block4.top()},
		    new bool[]{true});
path path4=flowpath(new pair[]{block3.bottom(), block5.left()},
		    new bool[]{false});
path path5=flowpath(new pair[]{block4.bottom(), block5.right()},
		    new bool[]{false});
                     
draw(block1);
draw(block2);
draw(block3);
draw(block4);
draw(block5);
drawflow(path1);
drawflow(path2,picture("Yes"),0.25,10*N);
drawflow(path3,picture("No"),0.25,10*N); 
drawflow(path4); 
drawflow(path5);

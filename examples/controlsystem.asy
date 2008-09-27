size(0,4cm);
import flowchart;

block delay=roundrectangle("$e^{-sT_t}$",(0.33,0));
block system=roundrectangle("$\frac{s+3}{s^2+0.3s+1}$",(0.6,0));
block controller=roundrectangle("$0.06\left( 1 + \frac{1}{s}\right)$",
				(0.45,-0.25));
block sum1=circle((0.15,0),mindiameter=0.3cm);
block junction1=circle((0.75,0),fillpen=currentpen);

draw(delay);
draw(system);
draw(controller);
draw(sum1);
draw(junction1);

add(new void(picture pic, transform t) {
    draw(pic,Label("$u$",0.5,N),path(new pair[]{t*(0,0),sum1.left(t)},
				     Horizontal),Arrow,PenMargin);

    draw(pic,path(new pair[]{sum1.right(t),delay.left(t)},Horizontal),Arrow,
	 PenMargin);
    label(pic,"-",sum1.bottom(t),ESE);

    draw(pic,path(new pair[]{delay.right(t),system.left(t)},Horizontal),Arrow,
	 PenMargin);

    draw(pic,path(new pair[]{system.right(t),junction1.left(t)},Horizontal),
	 PenMargin);

    draw(pic,Label("$y$",0.5,N),path(new pair[]{junction1.right(t),t*(0.9,0)},
				   Horizontal),Arrow,PenMargin);

    draw(pic,path(new pair[]{junction1.bottom(t),controller.right(t)},Vertical),
	 Arrow,PenMargin);

    draw(pic,path(new pair[]{controller.left(t),sum1.bottom(t)},Horizontal),
	 Arrow,PenMargin);
  });

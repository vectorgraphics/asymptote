size(0,4cm);
import flowchart;

block delay=roundrectangle("$e^{-sT_t}$",(0.33,0));
block system=roundrectangle("$\frac{s+3}{s^2+0.3s+1}$",(0.6,0));
block controller=roundrectangle("$0.06\left( 1 + \frac{1}{s}\right)$",
				(0.45,-0.25));
block sum1=circle("",(0.15,0),mindiameter=0.3cm);
block junction1=circle("",(0.75,0),fillpen=currentpen);

draw(delay);
draw(system);
draw(controller);
draw(sum1);
draw(junction1);

add(new void(picture pic, transform t) {
    blockconnector operator --=blockconnector(pic,t);
    
    block(0,0)--Label("$u$",align=N)--Arrow--sum1--Arrow--delay--Arrow--
      system--junction1--Label("$y$",align=N)--Arrow--block(1,0);

    junction1--Down--Left--Arrow--controller--Left--Up--
      Label("$-$",position=3,align=ESE)--Arrow--sum1;
  });

size(0,200);
real theta=40;
real phi=200;

pair O=(0,0); 
pair S=dir(theta);
pair R=(1,0);
pair P=rotate(phi)*S;
pair Q=rotate(phi)*R;

draw(circle(O,1.0));
draw(O--P--Q--cycle);
draw(O--R--S--cycle);

dot(O);
labeldot("$R$",R);
labeldot("$S$",S,dir(O--S));
labeldot("$P$",P,dir(O--P));
labeldot("$Q$",Q,dir(O--Q));
      
draw("$A-B$",arc(O,0.3,0,theta));
draw("$A-B$",rotate(phi)*arc(O,0.3,0,theta),1.2);

shipout();

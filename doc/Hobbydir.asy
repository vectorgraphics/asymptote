size(200);
pair z0=(0,0);
pair z1=(1,2);
pair z2=(2,1);

path g=z0..z1..z2;

label("$\ell_k$",z0--z1);
draw("$\ell_{k+1}$",z1--z2,dashed);
draw(z0--interp(z0,z1,1.5),dashed);
pair d1=dir(g,1);
draw(z1-d1..z1+d1,blue+dashed);
draw(g,blue);
draw(Label("$\theta_k$",0.4),arc(z1,0.4,degrees(z2-z1),degrees(d1)),blue,Arrow,
     EndPenMargin);
draw("$\phi_k$",arc(z1,0.4,degrees(d1),degrees(z1-z0),CCW),Arrow,
     EndPenMargin);

dot("$z_{k-1}$",z0,red);
dot("$z_k$",z1,NW,red);
dot("$z_{k+1}$",z2,red);

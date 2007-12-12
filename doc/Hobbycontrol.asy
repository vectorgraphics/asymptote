size(200);
pair z0=(0,0);
pair z1=(0.5,3);
pair z2=(2,1);

path g=z0..z1..z2;

pair d0=dir(g,0);
pair d1=dir(g,1);
draw(Label("$\omega_0$",1),z0-d0..z0+d0,blue+dashed,Arrow);
draw(Label("$\omega_1$",1),z1-d1..z1+1.5d1,blue+dashed,Arrow);
draw(z0--interp(z0,z1,1.5),dashed);
draw(subpath(g,0,1),blue);
draw("$\theta$",arc(z0,0.4,degrees(z1-z0),degrees(d0)),red,Arrow,
     EndPenMargin);
draw("$\phi$",arc(z1,1.05,degrees(z1-z0),degrees(d1)),red,Arrow,
     EndPenMargin);

dot("$z_0$",z0,SW,red);
dot("$z_1$",z1,SE,red);

size(10cm,0);
import math;

pair b=(0,0), c=(1,0);
pair a=extension(b,b+dir(60),c,c+dir(120));
pair d=extension(b,b+dir(30),a,a+dir(270));

draw(a--b--c--a--d--b^^d--c);
label("$A$",a,N);
label("$B$",b,W);
label("$C$",c,E);
label("$D$",d,S);

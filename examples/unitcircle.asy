size(0,150);

pair z0=0;
pair z1=1;
real theta=30;
pair z=dir(theta);

draw(circle(z0,1));
filldraw(z0--arc(z0,1,0,theta)--cycle,lightgrey);
dot(z0);
dot(Label,z1);
dot("$(x,y)=(\cos\theta,\sin\theta)$",z);
arrow("area $\frac{\theta}{2}$",dir(0.5*theta),2E);
draw("$\theta$",arc(z0,0.7,0,theta),LeftSide,Arrow,PenMargin);

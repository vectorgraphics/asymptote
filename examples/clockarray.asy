int nx=3;
int ny=4;
real xmargin=1cm;
real ymargin=xmargin;

size(settings.paperwidth,settings.paperheight);

picture pic;
real width=settings.paperwidth/nx-xmargin;
real height=settings.paperheight/ny-ymargin;
if(width <= 0 || height <= 0) abort("margin too big");
size(pic,width,height);

pen p=linewidth(0.5mm);
draw(pic,unitcircle,p);

real h=0.08;
real m=0.05;

for(int hour=1; hour <= 12; ++hour) {
  pair z=dir((12-hour+3)*30);
  label(pic,string(hour),z,z);
  draw(pic,z--(1-h)*z,p);
}

for(int minutes=0; minutes < 60; ++minutes) {
  pair z=dir(6*minutes);
  draw(pic,z--(1-m)*z);
}

dot(pic,(0,0));

frame f=pic.fit();
pair size=size(f)+(xmargin,ymargin);

for(int i=0; i < nx; ++i)
  for(int j=0; j < ny; ++j)
    add(shift(realmult(size,(i,j)))*f);


import fontsize;

settings.outformat="pdf";

defaultpen(Helvetica());

picture pic;
unitsize(pic,mm);

pair z=(0,0);
real length=88;
real height=8;
pair step=height*S;

label(pic,"Word Wall Spelling",z,Align);
z += step;
frame f;
label(f,"Name:");
pair z0=(max(f).x,min(f).y);
draw(f,z0--z0+50mm);
add(pic,f,z,Align);
z += step;

for(int i=1; i <= 15; ++i) {
  draw(pic,z--z+length);
  z += step;
  draw(pic,z--z+length,dashed+gray);
  z += step;
  void label(int i) {
    label(pic,string(i)+".",z,0.2NE,fontsize(0.8*1.5*2*height*mm)+gray);
  }
  if(i <= 10) label(i);
  else if(i == 11) {
    pair z0=z+length/2;
    pen p=fontsize(20pt);
    label(pic,"Challenge Word",z0+N*height,I*Align.y,p+basealign);
    label(pic,"(optional)",z0,I*Align.y,p);
  }
  else if(i == 12) label(1);
  else if(i == 13) label(2);
  else if(i == 14) label(3);
}
draw(pic,z--z+length);

add(pic.fit(),(0,0),W);
add(pic.fit(),(0,0),E);
newpage();
add(pic.fit(),(0,0),W);
add(pic.fit(),(0,0),E);



size(13cm,12cm,IgnoreAspect);
real minpercent=20;
real ignorebelow=0;
string data="diatom.csv";

currentpen=fontsize(8);
overwrite(MoveQuiet);

import graph;

file in=line(csv(input(data)));
pen p=linewidth(1);

string depthlabel=in;
string yearlabel=in;

string[] taxa=in;

real[] depth;
int[] year;
real[][] percentage;

while(true) {
  real d=in;
  if(eof(in)) break;
  depth.push(d);
  year.push(in);
  percentage.push(in);
}

percentage=transpose(percentage);
real depthmin=-min(depth);
real depthmax=-max(depth);

int n=percentage.length;

int final;
for(int taxon=0; taxon < n; ++taxon) {
  real[] P=percentage[taxon];
  if(max(P) < ignorebelow) continue;
  final=taxon;
}  

real location=0;
real bottom;
for(int taxon=0; taxon < n; ++taxon) {
  real[] P=percentage[taxon];
  real maxP=max(P);
  if(maxP < ignorebelow) continue;
  picture pic=new picture;
  real x=1;
  if(maxP < minpercent) x=minpercent/maxP;
  if(maxP > 100) x=50/maxP;
  scale(pic,Linear(x),Linear(false,-1));
  guide g=graph(pic,P,depth);
  draw(pic,g,p);
  xlimits(pic,0);
  crop(pic);
  filldraw(pic,(pic.userMin.x,depthmin)--g--(pic.userMin.x,depthmax)--cycle,
	   gray(0.9));
  xaxis(pic,0,Bottom,LeftTicks(false,0,2,"%.0f"));
  xaxis(pic,TeXify(taxa[taxon]),0.5,45,Top,NoTicks);
  if(taxon == 0) yaxis(pic,depthlabel,Left,RightTicks(0,10));
  if(taxon == final) yaxis(pic,Right,LeftTicks(0,10,""));
  
  add(shift(location,0)*pic);
  location += pic.userMax.x;
  bottom=pic.userMin.y;
}

for(int i=0; i < year.length; ++i)
  if(year[i] != 0) label((string) year[i],(location,-depth[i]),E);

label("\%",(0.5*location,bottom),5*S);

shipout();


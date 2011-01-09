import graph;

size(15cm,12cm,IgnoreAspect);

real minpercent=20;
real ignorebelow=0;
string data="diatom.csv";
string[] group;
int[] begin,end;

defaultpen(fontsize(8pt)+overwrite(MoveQuiet));

file in=input(data).line().csv();

string depthlabel=in;
string yearlabel=in;
string[] taxa=in;
group=in;
begin=in;
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

real angle=45;
real L=3cm;
pair Ldir=L*dir(angle);
real ymax=-infinity;
real margin=labelmargin();

real location=0;

for(int i=0; i < begin.length-1; ++i) end[i]=begin[i+1]-1;
end[begin.length-1]=n-1;

typedef void drawfcn(frame f);
drawfcn[] draw=new drawfcn[begin.length];

pair z0;

for(int taxon=0; taxon < n; ++taxon) {
  real[] P=percentage[taxon];
  real maxP=max(P);
  if(maxP < ignorebelow) continue;
  picture pic;
  real x=1;
  if(maxP < minpercent) x=minpercent/maxP;
  if(maxP > 100) x=50/maxP;
  scale(pic,Linear(true,x),Linear(-1));
  filldraw(pic,(0,depthmin)--graph(pic,P,depth)--(0,depthmax)--cycle,
           gray(0.9));
  xaxis(pic,Bottom,LeftTicks("$%.3g$",beginlabel=false,0,2),above=true);
  xaxis(pic,Top,above=true);

  frame label;
  label(label,rotate(angle)*TeXify(taxa[taxon]),(0,0),N);

  pair z=point(pic,N);
  pair v=max(label);
  int taxon=taxon;
  pic.add(new void(frame f, transform t) {
      pair z1=t*z+v;
      ymax=max(ymax,z1.y+margin);
    });

  for(int i=0; i < begin.length; ++i) {
    pair z=point(pic,N);
    pair v=max(label);
    if(taxon == begin[i]) {
      pic.add(new void(frame f, transform t) {
          pair Z=t*z+v;
          z0=Z;
          pair w0=Z+Ldir;
        });
    } else if(taxon == end[i]) {
      int i=i;
      pair align=2N;
      pic.add(new void(frame, transform t) {
          pair z0=z0;
          pair z1=t*z+v;
          pair w1=z1+Ldir;
          draw[i]=new void(frame f) {
            path g=z0--(z0.x+(ymax-z0.y)/Tan(angle),ymax)--
            (z1.x+(ymax-z1.y)/Tan(angle),ymax)--z1;
            draw(f,g);
            label(f,group[i],point(g,1.5),align);
          };
        });
    }
  }

  add(pic,label,point(pic,N));

  if(taxon == 0) yaxis(pic,depthlabel,Left,RightTicks(0,10),above=true);
  if(taxon == final) yaxis(pic,Right,LeftTicks("%",0,10),above=true);
 
  add(shift(location,0)*pic);
  location += pic.userMax().x;
}

add(new void(frame f, transform) {
    for(int i=0; i < draw.length; ++i)
      draw[i](f);
  });

for(int i=0; i < year.length; ++i)
  if(year[i] != 0) label((string) year[i],(location,-depth[i]),E);

label("\%",(0.5*location,point(S).y),5*S);

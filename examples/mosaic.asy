// Calendar example contributed by Jens Schwaiger

// transformations
path similarpath(pair a, pair b, path p) {
  // transform p into a path starting at a and ending at b
  pair first;
  pair last;
  path p_;
  first=point(p,0);
  last=point(p,length(p));
  p_=shift(-first)*p;
  p_=rotate(degrees(b-a))*p_;
  p_=scale(abs(b-a)/abs(last-first))*p_;
  p_=shift(a)*p_;
  return p_;
}

path c_line(path p) {
  // returns the path obtained by adding to p a copy rotated
  // around the endpoint of p by 180 degrees
  // works only if the initial point and the endpoint of p are different
  // a c_line is symetric with respect to the center of 
  // the straight line between its endpoints
  //
  return p..rotate(180,point(p,length(p)))*reverse(p);
}

path tounitcircle(path p, int n=300) {
  // the transformation pair x --> x/sqrt(1+abs(x)^2)
  // is a bijection from the plane to the open unitdisk
  real l=arclength(p);
  path ghlp;
  for(int i=0; i <= n; ++i) {
    real at=arctime(p,l/n*i);
    pair phlp=point(p,at);
    real trhlp=1/(1+abs(phlp)^2)^(1/2);
    ghlp=ghlp--trhlp*phlp;
  }
  if(cyclic(p)) {ghlp=ghlp--cycle;}
  return ghlp;
}

void centershade(picture pic=currentpicture, path p, pen in, pen out,
                 pen drawpen=currentpen) { 
  pair center=0.5(max(p)+min(p));
  real radius=0.5abs(max(p)-min(p));
  radialshade(pic,p,in,center,0,out,center,radius);
  draw(pic,p,drawpen);
}

pair zentrum(path p) {return 0.5(min(p)+max(p));}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

real scalefactor=19/13; // For output: height=scalefactor*width
real outputwidth=13cm;
picture kalender;// at first we produce a calendar for february 2006
texpreamble("\usepackage[latin1]{inputenc}");
size(outputwidth,0);
real yc=0.5;
pair diff=(-3.5,5*yc);
pen farbe(int j) {
  pen hlp=0.8white;
  if(j % 7 == 6) {hlp=red+white;}
  return hlp;}

// farbe=German word for color
path kasten=yscale(yc)*unitsquare;
// Kasten is a German word meaning something like box
path Gkasten=shift((0,2*yc)+diff)*xscale(7)*yscale(2)*kasten;
path tage[]= new path[7]; // Tag=day
string wochentag[]={"MO","DI","MI","DO","FR","SA","SO"};
path[][] bx= new path[6][7];
string[][] entry= new string[6][7];
bool[][] holiday=new bool[6][7];

// Now the necessary information for February 2006
int start=2;
int days=28;
for(int i=0; i < entry.length; ++i) {
  for(int j=0; j < entry[0].length; ++j) {
    int day=i*7+j-start+1;
    entry[i][j]=(day > 0 && day <= days ? (string) day : "");
    holiday[i][j]=false;
  }
}

for(int j=0; j < 7; ++j) {
  tage[j]=shift((j,yc)+diff)*kasten; 
  filldraw(tage[j],farbe(j),black+2bp);
  label(wochentag[j],zentrum(tage[j]),Palatino());
  for(int i=0; i < 6; ++i) {bx[i][j]=shift((j,-yc*i)+diff)*kasten;
    filldraw(bx[i][j],farbe(j),black+2bp);
    if(holiday[i][j]) {filldraw(bx[i][j],farbe(6),black+2bp);};
  };
};
filldraw(Gkasten,0.3white,black+2bp); 
for(int j=0; j < 7; ++j)
  for(int i=0; i < 6 ; ++i) {label(entry[i][j],zentrum(bx[i][j]),Palatino());}
label("\Huge Februar 2006",zentrum(Gkasten),Palatino()+white); 
// Zentrum=center; Februar=february
add(kalender,currentpicture);
erase();

// Now the mosaic is constructed
pair a[]=new pair[4];
path p[]=new path[4];
path q[]=new path[4];
path kontur[]=new path[5];
picture temppic;

a[1]=(0,0);
a[2]=(1,0);
a[3]=(0,1); // a triangle with abs(a[2]-a[1])=abs(a[3]-a[1]
            // and a right angle at a[1];
q[1]=(0,0){dir(-20)}::{dir(20)}(0.2,0){dir(-140)}..{dir(0)}(0.3,-0.2){dir(0)}..
{dir(140)}(0.4,0){dir(20)}..{dir(-20)}(1,0);
q[2]=(0,0){dir(20)}..{dir(-20)}(0.8,0){dir(-140)}..{dir(0)}(0.9,-0.3){dir(0)}..
{dir(140)}(1,0);
q[2]=c_line(q[2]);
p[1]=similarpath(a[1],a[2],q[1]);// arbitrary path from a[1] to a[2]
p[2]=similarpath(a[2],a[3],q[2]);// arbitrary c_line from a[2] to a[3]
p[3]=rotate(90,a[1])*reverse(p[1]);//
kontur[1]=p[1]..p[2]..p[3]..cycle;// first tile
kontur[2]=rotate(90,a[1])*kontur[1];// second
kontur[3]=rotate(180,a[1])*kontur[1];// third
kontur[4]=rotate(270,a[1])*kontur[1];// fourth
pair tri=2*(interp(a[2],a[3],0.5)-a[1]);
pair trii=rotate(90)*tri;
// translations of kontur[i], i=1,2,3,4, with respect to
// j*tri+k*trii
// fill the plane

for(int j=-4; j < 4; ++j)
  for(int k=-4; k < 4; ++k) {
    transform tr=shift(j*tri+k*trii);
    for(int i=1; i < 5; ++i) {
      centershade(temppic,tr*kontur[i],(1-i/10)*white,
                  (1-i/10)*chartreuse,black+2bp);
    }
  }
  
// Now we produce the bijective images inside 
// a suitably scaled unitcircle            
for(int k=-1; k < 2; ++k)
  for(int l=-1; l < 2; ++l) {
    transform tr=shift(k*tri+l*trii);
    for(int i=1; i < 5; ++i) {
      centershade(temppic,scale(2.5)*tounitcircle(tr*kontur[i],380),
                  (1-i/10)*white,(1-i/10)*orange,black+2bp);
    }
  }         
          
add(temppic); 

// We clip the picture to a suitable box 
pair piccenter=0.5*(temppic.min()+temppic.max());
pair picbox=temppic.max()-temppic.min();
real picwidth=picbox.x;
transform trialtrans=shift(0,-1.5)*shift(piccenter)*yscale(scalefactor)*
  scale(0.25picwidth)*shift((-0.5,-0.5))*identity();
clip(trialtrans*unitsquare);

// add the calendar at a suitable position
add(kalender.fit(0.75*outputwidth),interp(point(S),point(N),1/13)); 

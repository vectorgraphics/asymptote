size(20cm);

// The required data file is available here:
// http://www.uni-graz.at/~schwaige/asymptote/worldmap.dat
// This data was originally obtained from
// http://www.ngdc.noaa.gov/mgg_coastline/mapit.jsp 

real findtheta(real phi, real epsilon=realEpsilon) {
  // Determine for given phi the unique solution -pi/2 <= theta <= pi/2 off
  // 2*theta+sin(2*theta)=pi*sin(phi)
  // in the non-trivial cases by Newton iteration;
  // theoretically the initial guess pi*sin(phi)/4  always works.
  real nwtn(real x, real y) {return x-(2x+sin(2x)-y)/(2+2*cos(2x));};
  real y=pi*sin(phi);
  if(y == 0) return 0.0;
  if(abs(y) == 1) return pi/2;
  real startv=y/4;
  real endv=nwtn(startv,y);
  if(epsilon < 500*realEpsilon) epsilon=500*realEpsilon;
  while(abs(endv-startv) > epsilon) {startv=endv; endv=nwtn(startv,y);};
  return endv;
}

pair mollweide(real lambda, real phi, real lambda0=0){
  // calculate the Mollweide projection centered at lambda0 for the point
  // with coordinates(phi,lambda) 
  static real c1=2*sqrt(2)/pi;
  static real c2=sqrt(2);
  real theta=findtheta(phi);
  return(c1*(lambda-lambda0)*cos(theta), c2*sin(theta));
} 	

guide gfrompairs(pair[] data){
  guide gtmp;
  for(int i=0; i < data.length; ++i) {
    pair tmp=mollweide(radians(data[i].y),radians(data[i].x));
    gtmp=gtmp--tmp;
  }
  return gtmp;
}

string datafile="worldmap.dat";

file in=input(datafile,comment="/").line();
// new commentchar since "#" is contained in the file
pair[][] arrarrpair=new pair[][] ;
int cnt=-1;
bool newseg=false;
while(true) {
  if(eof(in)) break;
  string str=in;
  string[] spstr=split(str,"");

  if(spstr[0] == "#") {++cnt; arrarrpair[cnt]=new pair[] ; newseg=true;}
  if(spstr[0] != "#" && newseg) {
    string[] spstr1=split(str,'\t'); // separator is TAB not SPACE
    pair tmp=((real) spstr1[1],(real) spstr1[0]); 
    arrarrpair[cnt].push(tmp);
  }
}

for(int i=0; i < arrarrpair.length; ++i)
  draw(gfrompairs(arrarrpair[i]),1bp+black);

// lines of longitude and latitude
pair[] constlong(real lambda, int np=100) {
  pair[] tmp;
  for(int i=0; i <= np; ++i) tmp.push((-90+i*180/np,lambda));
  return tmp;
}

pair[] constlat(real phi, int np=100) {
  pair[] tmp;
  for(int i=0; i <= 2*np; ++i) tmp.push((phi,-180+i*180/np));
  return tmp;
}

for(int j=1; j <= 5; ++j) draw(gfrompairs(constlong(-180+j/6*360)),white);
draw(gfrompairs(constlong(-180)),1.5bp+white);
draw(gfrompairs(constlong(180)),1.5bp+white);
for(int j=0; j <= 12; ++j) draw(gfrompairs(constlat(-90+j/6*180)),white); 	
//draw(gfrompairs(constlong(10)),dotted);

close(in);
shipout(bbox(1mm,darkblue,Fill(lightblue)), view=true);


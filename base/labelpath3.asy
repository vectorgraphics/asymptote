// Fit a label to a path3.
// Author: Jens Schwaiger

import three;
private real eps=100*realEpsilon;

triple nextnormal(triple p, triple q)
{
  triple nw=p-(dot(p,q)*q);
  return abs(nw) < 0.0001 ? p : unit(nw);
}

triple[] firstframe(path3 p, triple optional=O)
{
  triple[] start=new triple[3];
  start[0]=dir(p,reltime(p,0));
  start[1]=(abs(cross(start[0],optional)) < eps) ? perp(start[0]) :
    unit(cross(start[0],optional));
  start[2]=cross(start[0],start[1]);
  return start;
}

// Modification of the bishop frame construction contained in
// space_tube.asy (from Philippe Ivaldi's modules). 
// For noncyclic path3s only
triple[] nextframe(path3 p, real reltimestart, triple[] start, real
                   reltimeend, int subdiv=20)
{
  triple[][] bf=new triple[subdiv+1][3];
  real lg=reltimeend-reltimestart;
  if(lg <= 0) return start;
  bf[0]=start;
  int n=subdiv+1;
  for(int i=1; i < n; ++i)
    bf[i][0]=dir(p,reltime(p,reltimestart+(i/subdiv)*lg));

  for(int i=1; i < n; ++i) {
    bf[i][1]=nextnormal(bf[i-1][1],bf[i][0]);
    bf[i][2]=cross(bf[i][0],bf[i][1]);
  }
  return bf[subdiv];
}
  
surface labelpath(string s, path3 p, real angle=90, triple optional=O)
{
  real Cos=Cos(angle);
  real Sin=Sin(angle);
  path[] text=texpath(Label(s,(0,0),Align,basealign));
  text=scale(1/(max(text).x-min(text).x))*text;
  path[][] decompose=containmentTree(text);
        
  real[][] xpos=new real[decompose.length][2];
  surface sf;
  for(int i=0; i < decompose.length; ++i) {// Identify positions along x-axis   
    xpos[i][1]=i;
    real pos0=0.5(max(decompose[i]).x+min(decompose[i]).x);
    xpos[i][0]=pos0;
  }
  xpos=sort(xpos); // sort by distance from 0;
  triple[] pos=new triple[decompose.length];
  real lg=arclength(p);
  //create frames;
  triple[] first=firstframe(p,optional);
  triple[] t0=first;
  real tm0=0;
  triple[][] bfr=new triple[decompose.length][3];
  for(int j=0; j < decompose.length; ++j) {
    bfr[j]=nextframe(p,tm0,t0,xpos[j][0]);
    tm0=xpos[j][0]; t0=bfr[j];
  }
  transform3[] mt=new transform3[bfr.length];
  for(int j=0; j < bfr.length; ++j) {
    triple f2=Cos*bfr[j][1]+Sin*bfr[j][2];
    triple f3=Sin*bfr[j][1]+Cos*bfr[j][2];
    mt[j]=shift(relpoint(p,xpos[j][0]))*transform3(bfr[j][0],f2,f3);
  }
  for(int j=0; j < bfr.length; ++j) {
    path[] dc=decompose[(int) xpos[j][1]];
    pair pos0=(0.5(max(dc).x+min(dc).x),0);
    sf.append(mt[j]*surface(scale(lg)*shift(-pos0)*dc));
  }
  return sf;
}

// approximate acceleration
triple accel(path3 g, real t)
{
  triple a;
  real deltaT=0.1;
  if(t < deltaT) t=deltaT;
  if(t > 1-deltaT) t=1-deltaT;
  a=dir(g,t+deltaT)-dir(g,t-deltaT);
  return a/deltaT/2;
}

// transformation that bends points along a path
// assumes that p.z is in [0,scale]
triple bend(triple p, path3 g, real scale=1, real endtime=g.length())
{
  real time=arctime(g,arclength(subpath(g,0,endtime))+(p.z-scale));
  triple dir=dir(g,time);

  triple w=point(g,time);

  triple dir2=unit(cross(dir(g,0),dir(g,1)));
  if(abs(dir2) < 1000*realEpsilon) {
    if(abs(dir-X) < 1000*realEpsilon || abs(dir+X) < 1000*realEpsilon)
      dir2=Y;
    else dir2=X;
  }
  dir2=unit(dir2-dot(dir2,dir)*dir);

  triple w=cross(dir,dir2);
  triple q=point(g,time);
  transform3 t=new real[][] {
    {w.x,dir2.x,dir.x,0},
    {w.y,dir2.y,dir.y,0},
    {w.z,dir2.z,dir.z,0},
    {0,0,0,1}
  };
  return shift(q-t*(0,0,p.z))*t*p;
}

real takeStep(path3 s, real endtime, real width)
{
  real newend=1;
  if(!s.straight(0)) { // use a better test ? 
    real a=abs(accel(s,endtime));

    // tweak this:
    real K=0.85/width; // a different constant perhaps ?
    real minStep=1/50; // at most 1/minStep segments for a curve
    real step=max(1/(K*a+1),minStep); // or a different model
    newend=min(endtime+step,length(s));
    a=abs(accel(s,(newend+endtime)/2));
    step=max(1/(K*a+1),minStep);
    if(step > 0.25) step=0.25; // guarantee at least 4 segments per curve
    newend=min(endtime+step,length(s));
  }
  return newend;
}

surface tube(path3 g, real width)
{
  surface tube;
  real r=0.5*width;
  
  transform3 t=scale3(r);

  for(int i=0; i < length(g); ++i) {
    if(straight(g,i)) {
      triple v=point(g,i);
      triple u=point(g,i+1)-v;
      tube.append(shift(v)*transform3(unit(u))*scale(r,r,abs(u))*
		  unitcylinder);
    } else {
      path3 s=subpath(g,i,i+1);
      real endtime=0;
      while(endtime < 1) {
	real newend=takeStep(s,endtime,r);

	path3 si=subpath(s,endtime,newend);

	real L=arclength(si);
	surface segment=scale(r,r,L)*unitcylinder;
	path3 circle=t*unitcircle3;
	if(endtime == 0)
	  segment.push(circle);
	if(newend == 1)
	  segment.push(shift((0,0,L))*circle);
	for(patch p : segment.s) {
	  for(int i=0; i < 4; ++i) {
	    for(int j=0; j < 4; ++j) {
	      p.P[i][j]=bend(p.P[i][j],si,L,1);
	    }
	  }
	}
	tube.s.append(segment.s);

	endtime=newend;
      }
    }

    if(i > 0 && abs(dir(g,i,1)-dir(g,i,-1)) > 100*realEpsilon)
      tube.append(shift(point(g,i))*t*unitsphere);
  }
  return tube;
}

struct arrowhead3
{
  surface head(path3 g, position position, pen p=currentpen,
               real size=0, real angle=arrowangle);
  real size(pen p)=arrowsize;
}

arrowhead3 DefaultHead3;

DefaultHead3.head=new surface(path3 g, position position, pen p=currentpen,
                              real size=0, real angle=arrowangle) {
  if(size == 0) size=DefaultHead3.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 s=subpath(g,arctime(g,arclength(g)-size),length(g));
  real L=arclength(s);
  real wl=Tan(angle);
  real width=L*wl; // make sure this is not zero
  real remainL=L;

  surface arrow;

  for(int i=0; i < length(s); ++i) {
    path3 s=subpath(s,i,i+1);

    real endtime=0;
    while(endtime < 1) {
      real newend=takeStep(s,endtime,width);

      path3 si=subpath(s,endtime,newend);

      real l=arclength(si);
      real w=remainL*wl;
      surface segment=scale(w,w,l)*unitcylinder;
      if(endtime == 0) // add base
        segment.push(scale3(w)*unitcircle3);
      for(patch p : segment.s) {
        for(int i=0; i < 4; ++i) {
          for(int j=0; j < 4; ++j) {
            real k=1-p.P[i][j].z/remainL;
            p.P[i][j]=(k*p.P[i][j].x,k*p.P[i][j].y,p.P[i][j].z);
            p.P[i][j]=bend(p.P[i][j],si,l,1);
          }
        }
      }
      arrow.s.append(segment.s);

      endtime=newend;
      remainL -= l;
    }
  }

  return arrow;
};

void arrow(picture pic, arrowhead3 arrowhead=DefaultHead3,
           path3 g, pen p=currentpen, pen arrowheadpen=p, real size=0,
           real angle=arrowangle, position position=EndPoint,
           bool forwards=true, bool center=false)
{
  if(size == 0) size=arrowhead.size(arrowheadpen);
  size=min(arrowsizelimit*arclength(g),size);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) {
    position *= arclength(g);
    if(center) position += 0.5*size;
    position=arctime(g,position);
  } else if(center) 
    position=arctime(g,arclength(subpath(g,0,position))+0.5*size);
  int L=length(g);
  if(!forwards) {
    g=reverse(g);
    position=L-position;
  }

  path3 r=subpath(g,0,position);
  path3 s=subpath(g,position,L);

  size=min(arrowsizelimit*arclength(r),size);

  if(opacity(arrowheadpen) < 1 || position == L) {
    draw(pic,subpath(r,0,arctime(r,arclength(r)-size)),p);
    if(position < L) draw(pic,s,p);
  } else draw(pic,g,p);
  if(arclength(r) > size)
    draw(pic,arrowhead.head(r,position,p,size,angle),arrowheadpen);
}

picture arrow(arrowhead3 arrowhead=DefaultHead3,
              path3 g, pen p=currentpen, pen arrowheadpen=p, real size=0,
              real angle=arrowangle, position position=EndPoint,
              bool forwards=true, bool center=false)
{
  picture pic;
  pic.add(new void(picture f, transform3 t) {
      arrow(f,arrowhead,t*g,p,arrowheadpen,size,angle,position,
            forwards,center);
    });

  if(size == 0) size=DefaultHead3.size(p);
  bool relative=position.relative;
  real position=position.position.x;
  if(relative) position=reltime(g,position);

  path3 r=subpath(g,0,position);
  if(arclength(r) > size) {
    surface s=arrowhead.head(r,position,p,size,angle),arrowheadpen;
    triple v=point(r,length(r));
    pic.addPoint(v,min(s)-v);
    pic.addPoint(v,max(s)-v);
  }
  addPath(pic,g,p);
  return pic;
}

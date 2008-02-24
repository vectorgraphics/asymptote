import graph_settings;
real stepfraction=0.05;

picture slopefield(real f(real,real), pair a, pair b,
                   int nx=nmesh, int ny=nx,
                   real tickfactor=0.5, pen p=currentpen, arrowbar arrow=None)
{
  picture pic;
  real dx=(b.x-a.x)/nx;
  real dy=(b.y-a.y)/ny;
  real step=0.5*tickfactor*min(dx,dy);

  for(int i=0; i <= nx; ++i) {
    real x=a.x+i*dx;
    for(int j=0; j <= ny; ++j) {
      pair cp=(x,a.y+j*dy);
      real slope=f(cp.x,cp.y);
      real mp=step/sqrt(1+slope^2);
      draw(pic,(cp.x-mp,cp.y-mp*slope)--(cp.x+mp,cp.y+mp*slope),p,arrow); 
    }
  }
  clip(pic,box(a,b));
  return pic;
}

picture slopefield(real f(real), pair a, pair b,
                   int nx=nmesh, int ny=nx, pen p=currentpen,
		   arrowbar arrow=None)
{
  return slopefield(new real(real x, real y) {return f(x);},a,b,nx,ny,p,arrow);
}

path curve(pair c, real f(real,real), pair a, pair b) 
{
  real step=stepfraction*(b.x-a.x);     
  real halfstep=0.5*step;
  real sixthstep=step/6;
        
  path follow(real sign) {
    pair cp=c;
    guide g=cp;
    real dx,dy;
    real factor=1;
    do {
      real slope;
      pair S(pair z) {
        slope=f(z.x,z.y);
        return factor*sign/sqrt(1+slope^2)*(1,slope);
      }
      pair S3;
      pair advance() {
        pair S0=S(cp);
        pair S1=S(cp+halfstep*S0);
        pair S2=S(cp+halfstep*S1);
        S3=S(cp+step*S2);
        pair cp0=cp+sixthstep*(S0+2S1+2S2+S3);
        dx=min(cp0.x-a.x,b.x-cp0.x);
        dy=min(cp0.y-a.y,b.y-cp0.y);
        return cp0;
      }
      pair cp0=advance();
      if(dx < 0) {
        factor=(step+dx)/step;
        cp0=advance();
        g=g..{S3}cp0{S3};
        break;
      }
      if(dy < 0) {
        factor=(step+dy)/step;
        cp0=advance();
        g=g..{S3}cp0{S3};
        break;
      }
      cp=cp0;
      g=g..{S3}cp{S3};
    } while (dx > 0 && dy > 0);
    return g;
  }

  return reverse(follow(-1))&follow(1);
}

path curve(pair c, real f(real), pair a, pair b)
{
  return curve(c,new real(real x, real y){return f(x);},a,b);
}


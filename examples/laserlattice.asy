import graph;
import palette;

int n=256;
pen[] Palette=BWRainbow();

real w(real w0, real z0, real z) {return w0*sqrt(1+(z/z0)^2);}

real pot(real lambda, real w0, real r, real z)
{
  real z0=pi*w0^2/lambda, kappa=2pi/lambda;
  return exp(-2*(r/w(w0,z0,z))^2)*cos(kappa*z)^2;
}

picture make_field(real lambda, real w0)
{
  real[][] v=new real[n][n];
  for(int i=0; i < n; ++i)
    for(int j=0; j < n; ++j)
      v[i][j]=pot(lambda,w0,i-n/2,abs(j-n/2));

  picture p=new picture;
  size(p,250,250,IgnoreAspect);
  real xm=-n/lambda, ym=-n/(2*w0), xx=n/lambda, yx=n/(2*w0);
  image(p,v,(xm,ym),(xx,yx),Palette);
  xlimits(p,xm,xx);
  ylimits(p,ym,yx);
  xaxis(p,"{\Large $z/\frac{\lambda}{2}$}",BottomTop,LeftTicks);
  yaxis(p,"{\Large $r/w_0$}",LeftRight,RightTicks);
  label(p,format("{\LARGE $w_0/\lambda=%.2f$}",w0/lambda),point(p,NW),5N);

  return p;
}

picture p=make_field(160,80);
picture q=make_field(80,80);
picture r=make_field(16,80);
picture s=make_field(2,80);

real margin=1cm;
add(p.fit(),(0,0),margin*NW);
add(q.fit(),(0,0),margin*NE);
add(r.fit(),(0,0),margin*SW);
add(s.fit(),(0,0),margin*SE);

path defaultshading(real s0=0.03, real s1=0.25, real s2=1-s1, real s3=1-s0)
{
  return (0,s0)..controls (1/3,s1) and (2/3,s2)..(1,s3);
}

path defaultshading=defaultshading();

typedef real shadefcn(real x);

real defaultshade(real x) {
  return point(defaultshading,x).y;
}

struct light {
  triple source;
  shadefcn shade;
  bool on=false;
  
  void operator init(triple source, shadefcn shade=defaultshade) {
    this.source=unit(source);
    this.shade=shade;
    on=true;
  }

  real intensity(triple v) {
    return on ? shade(0.5(dot(unit(v),source)+1)) : 1;
  }
}

light operator cast(triple v) {return light(v);}

light currentlight=(0.25,-0.25,1);
light nolight;

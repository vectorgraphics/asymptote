import three;

path defaultshading(real s0=0.03, real s1=0.25, real s2=1-s1, real s3=1-s0)
{
  return (0,s0)..controls (1/3,s1) and (2/3,s2)..(1,s3);
}

public path defaultshading=defaultshading();

typedef real shadefcn(real x);

real defaultshade(real x) {
  return point(defaultshading,x).y;
}

struct light {
  public triple source;
  public shadefcn shade;
  
  static light init(triple source, shadefcn shade=defaultshade) {
    light L=new light;
    L.source=source;
    L.shade=shade;
    return L;
  }
  
  static light init(real x, real y , real z, shadefcn shade=defaultshade) {
    return init((x,y,z),shade);
  }
    
  real intensity(triple v) {
    return shade(0.5(dot(v,source)+1));
  }
}

light currentlight=light.init(0.25,-0.25,1);

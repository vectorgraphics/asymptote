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

struct material {
  pen[] p; // surfacepen,ambientpen,emissivepen,specularpen
  real opacity;
  real shininess;  
  real granularity;
  void operator init(pen surfacepen=lightgray, pen ambientpen=black,
		     pen emissivepen=black, pen specularpen=mediumgray,
		     real opacity=opacity(surfacepen),
		     real shininess=defaultshininess,
		     real granularity=-1) {
    p=new pen[] {surfacepen,ambientpen,emissivepen,specularpen};
    this.opacity=opacity;
    this.shininess=shininess;
    this.granularity=granularity;
  }
  void operator init(material m) {
    p=copy(m.p);
    opacity=m.opacity;
    shininess=m.shininess;
    granularity=m.granularity;
  }
}

bool operator == (material m, material n)
{
  return all(m.p == n.p) && m.opacity == n.opacity &&
  m.shininess == n.shininess && m.granularity == n.granularity;
}

material operator cast(pen p)
{
  return material(p);
}

pen operator ecast(material m)
{
  return m.p.length > 0 ? m.p[0] : nullpen;
}

material emissive(pen p, real granularity=0)
{
  return material(black,black,p,black,opacity(p),1,granularity);
}

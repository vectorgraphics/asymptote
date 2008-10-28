struct material {
  pen[] p; // diffusepen,ambientpen,emissivepen,specularpen
  real opacity;
  real shininess;  
  real granularity;
  void operator init(pen diffusepen=black, pen ambientpen=black,
                     pen emissivepen=black, pen specularpen=mediumgray,
                     real opacity=opacity(diffusepen),
                     real shininess=defaultshininess,
                     real granularity=-1) {
    p=new pen[] {diffusepen,ambientpen,emissivepen,specularpen};
    this.opacity=opacity;
    this.shininess=shininess;
    this.granularity=granularity;
  }
  void operator init(material m, real granularity=m.granularity) {
    p=copy(m.p);
    opacity=m.opacity;
    shininess=m.shininess;
    this.granularity=granularity;
  }
  pen diffuse() {return p[0];}
  pen ambient() {return p[1];}
  pen emissive() {return p[2];}
  pen specular() {return p[3];}
  void diffuse(pen q) {p[0]=q;}
  void ambient(pen q) {p[1]=q;}
  void emissive(pen q) {p[2]=q;}
  void specular(pen q) {p[3]=q;}
}

void write(file file, string s="", material x, suffix suffix=none)
{
  write(file,s);
  write(file,"{");
  write(file,"diffuse=",x.diffuse());
  write(file,", ambient=",x.ambient());
  write(file,", emissive=",x.emissive());
  write(file,", specular=",x.specular());
  write(file,", opacity=",x.opacity);
  write(file,", shininess=",x.shininess);
  write(file,", granularity=",x.granularity);
  write(file,"}",suffix);
}

void write(string s="", material x, suffix suffix=endl)
{
  write(stdout,s,x,suffix);
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

material[] operator cast(pen[] p)
{
  return sequence(new material(int i) {return p[i];},p.length);
}

pen operator ecast(material m)
{
  return m.p.length > 0 ? m.diffuse() : nullpen;
}

material emissive(material m, real granularity=m.granularity)
{
  return material(black+opacity(m.opacity),black,m.diffuse(),black,m.opacity,1,
                  granularity);
}

real[] rgba(pen p)
{
  real[] a=colors(rgb(p));
  a.push(opacity(p));
  return a;
}

pen rgba(real[] a)
{
  return rgb(a[0],a[1],a[2])+opacity(a[3]);
}

struct light {
  real[][] diffuse;
  real[][] ambient;
  real[][] specular;
  real specularfactor;
  bool viewport; // Are the lights specified (and fixed) in the viewport frame?
  triple[] position; // Only directional lights are implemented.

  transform3 T=identity(4); // Transform to apply to normal vectors.

  bool on() {return position.length > 0;}
  
  void operator init(pen[] diffuse=array(position.length,white),
		     pen[] ambient=array(position.length,black),
		     pen[] specular=diffuse, real specularfactor=1,
		     bool viewport=true, triple[] position) {
    this.position=new triple[position.length];
    this.diffuse=new real[position.length][];
    this.ambient=new real[position.length][];
    this.specular=new real[position.length][];
    for(int i=0; i < position.length; ++i) {
      this.position[i]=unit(position[i]);
      this.diffuse[i]=rgba(diffuse[i]);
      this.ambient[i]=rgba(ambient[i]);
      this.specular[i]=rgba(specular[i]);
    }
    this.specularfactor=specularfactor;
    this.viewport=viewport;
  }

  void operator init(pen diffuse=white, pen ambient=black, pen specular=diffuse,
		     real specularfactor=1,
		     bool viewport=true...triple[] position) {
    int n=position.length;
    operator init(array(n,diffuse),array(n,ambient),array(n,specular),
		  specularfactor,viewport,position);
  }

  void operator init(pen diffuse=white, pen ambient=black, pen specular=diffuse,
		     bool viewport=true, real x, real y, real z) {
    operator init(diffuse,ambient,specular,viewport,(x,y,z));
  }

  void operator init(explicit light light) {
    diffuse=copy(light.diffuse);
    ambient=copy(light.ambient);
    specular=copy(light.specular);
    specularfactor=light.specularfactor;
    viewport=light.viewport;
    position=copy(light.position);
  }

  pen color(triple normal, material m, transform3 T=T) {
    if(position.length == 0) return m.diffuse();
    normal=T*normal;
    normal=unit(normal)*sgn(normal.z);
    real s=m.shininess*128;
    real[] Diffuse=rgba(m.diffuse());
    real[] Ambient=rgba(m.ambient());
    real[] Specular=rgba(m.specular());
    real[] p=rgba(m.emissive());
    for(int i=0; i < position.length; ++i) {
      triple L=viewport ? position[i] : T*position[i];
      real Ldotn=max(dot(normal,L),0);
      p += ambient[i]*Ambient+Ldotn*diffuse[i]*Diffuse;
// Apply specularfactor to partially compensate non-pixel-based rendering.
      if(Ldotn > 0) // Phong-Blinn model of specular reflection
	p += dot(normal,unit(L+Z))^s*specularfactor*specular[i]*Specular;
    }
    return rgb(p[0],p[1],p[2])+opacity(opacity(m.diffuse()));
  }
}

light operator * (transform3 t, light light)
{
  light light=light(light);
  if(!light.viewport) light.position=shiftless(t)*light.position;
  return light;
}

light operator cast(triple v) {return light(v);}

light currentlight=light(ambient=rgb(0.1,0.1,0.1),(0.25,-0.25,1),
			 specularfactor=3);

light adobe=light(gray(0.4),specularfactor=3,viewport=false,
		  (0.5,-0.5,-0.25),(0.5,0.5,0.25),
		  (0.5,-0.5,0.2),(-0.5,0.5,-0.2));
light nolight;

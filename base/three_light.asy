struct material {
  pen[] p; // diffusepen,ambientpen,emissivepen,specularpen
  real opacity;
  real shininess;  
  real granularity;
  void operator init(pen diffusepen=lightgray, pen ambientpen=black,
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

pen operator ecast(material m)
{
  return m.p.length > 0 ? m.diffuse() : nullpen;
}

material emissive(material m, real granularity=m.granularity)
{
  return material(black+opacity(m.opacity),black,m.diffuse(),black,m.opacity,1,
                  granularity);
}

pen pack(real[] p) 
{
  return rgb(p[0],p[1],p[2])+opacity(p[3]);
}

real[] unpack(pen p)
{
  real[] a=colors(rgb(p));
  a.push(opacity(p));
  return a;
}

struct light {
  triple[] position;
  real[][] diffuse;
  real[][] ambient;
  real[][] specular;
  real specularfactor;
  bool viewport; // Are the lights specified (and fixed) in the viewport frame?
  transform3 T=identity(4); // Transform to apply to normal vectors.

  bool on() {return position.length > 0;}
  
  void operator init(triple[] position,
		     pen[] diffuse=array(position.length,white),
		     pen[] ambient=array(position.length,black),
		     pen[] specular=diffuse, real specularfactor=1,
		     bool viewport=true) {
    this.position=new triple[position.length];
    this.diffuse=new real[position.length][];
    this.ambient=new real[position.length][];
    this.specular=new real[position.length][];
    for(int i=0; i < position.length; ++i) {
      this.position[i]=unit(position[i]);
      this.diffuse[i]=unpack(diffuse[i]);
      this.ambient[i]=unpack(ambient[i]);
      this.specular[i]=unpack(specular[i]);
    }
    this.specularfactor=specularfactor;
    this.viewport=viewport;
  }

  void operator init(triple position, pen diffuse=white,
		     pen ambient=black, pen specular=diffuse,
		     real specularfactor=1, bool viewport=true) {
    operator init(new triple[] {position},new pen[] {diffuse},
		  new pen[] {ambient},new pen[] {specular},specularfactor,
		  viewport);
  }

  void operator init(real x, real y, real z,
		     real specularfactor=1, bool viewport=true) {
    operator init((x,y,z),specularfactor,viewport);
  }

  triple[] position(transform3 T) {
    return sequence(new triple(int i) {return T*position[i];},position.length);
  }

  pen color(triple normal, material m, transform3 T=T) {
    if(position.length == 0) return m.diffuse();
    normal=T*normal;
    normal=unit(normal)*sgn(normal.z);
    real s=m.shininess*128;
    real[] Diffuse=unpack(m.diffuse());
    real[] Ambient=unpack(m.ambient());
    real[] Specular=unpack(m.specular());
    real[] p=unpack(m.emissive());
    for(int i=0; i < position.length; ++i) {
      triple L=viewport ? position[i] : T*position[i];
      real Ldotn=max(dot(normal,L),0);
      p += ambient[i]*Ambient+Ldotn*diffuse[i]*Diffuse;
// Apply a factor of 3 to partially account for the non-pixel-based rendering.
      if(Ldotn > 0) // Phong-Blinn model of specular reflection
	p += dot(normal,unit(L+Z))^s*specularfactor*specular[i]*Specular;
    }
    return pack(p);
  }
}

light operator cast(triple v) {return light(v);}

light currentlight=light((0.25,-0.25,1),white,ambient=rgb(0.1,0.1,0.1),
			 specularfactor=3,viewport=true);

light adobe=light(new triple[] {(0,1,-0.25),(0,-1,-0.25),
				      (0.5,0,0.5),(-0.5,0,-0.5)},
  array(4,gray(0.45)),specularfactor=3,viewport=false);

light nolight;

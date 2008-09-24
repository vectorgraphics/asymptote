real specularfactor=10.0;

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

struct light {
  triple[] position;
  real[][] diffuse;
  real[][] ambient;
  real[][] specular;
  transform3 T=identity(4); // Temporary variable for passing modelview matrix

  bool on() {return position.length > 0;}
  
  void operator init(triple[] position, pen[] diffuse, pen[] ambient,
		     pen[] specular) {
    this.position=new triple[position.length];
    this.diffuse=new real[position.length][];
    this.ambient=new real[position.length][];
    this.specular=new real[position.length][];
    for(int i=0; i < position.length; ++i) {
      this.position[i]=unit(position[i]);
      this.diffuse[i]=colors(rgb(diffuse[i]));
      this.ambient[i]=colors(rgb(ambient[i]));
      this.specular[i]=colors(rgb(specular[i]));
    }
  }

  void operator init(triple position, pen diffuse=rgb(white), pen ambient=rgb(black),
		     pen specular=rgb(white)) {
    operator init(new triple[] {position}, new pen[] {diffuse},
		     new pen[] {ambient}, new pen[] {specular});
  }

  void operator init(real x, real y, real z) {
    operator init((x,y,z));
  }

  pen color(triple normal, pen p) {
    if(position.length == 0) return p;
    triple n=unit(normal)*sgn(normal.z);
    real[] Diffuse=colors(rgb(p));
    real[] p={0,0,0};
    for(int i=0; i < position.length; ++i)
      p += max(dot(n,position[i]),0)*diffuse[i]*Diffuse;
    return rgb(p[0],p[1],p[2]);
  }

  pen color(triple normal, triple eye, material m) {
    triple n=unit(normal)*sgn(normal.z);
    triple eye=unit(eye);
    real s=m.shininess*128;
    real[] Diffuse=colors(rgb(m.diffuse()));
    real[] Ambient=colors(rgb(m.ambient()));
    real[] Specular=colors(rgb(m.specular()));
    real[] p=colors(rgb(m.emissive()));
    for(int i=0; i < position.length; ++i) {
      triple L=position[i];
      p += ambient[i]*Ambient+max(dot(n,L),0)*diffuse[i]*Diffuse+
	specularfactor*(
	  max(dot(unit(2*n*dot(n,L)-L),eye),0)^s*specular[i]*Specular // Phong
//	  max(dot(n,unit(L+eye)),0)^s*specular[i]*Specular // Phong-Blinn
	  );
    }
    return rgb(p[0],p[1],p[2]);
  }
}

light operator cast(triple v) {return light(v);}

light currentlight=(0.25,-0.25,1);
currentlight.ambient[0]=new real[]{0.1,0.1,0.1};

light nolight;

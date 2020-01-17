struct material {
  pen[] p; // diffusepen,emissivepen,specularpen
  real opacity;
  real shininess;  
  real metallic;
  real fresnel0; // Reflectance rate at a perfect normal angle.

  void operator init(pen diffusepen=black,
                     pen emissivepen=black, pen specularpen=mediumgray,
                     real opacity=opacity(diffusepen),
                     real shininess=defaultshininess,
                     real metallic=defaultmetallic,
                     real fresnel0=defaultfresnel0) {

    p=new pen[] {diffusepen,emissivepen,specularpen};
    this.opacity=opacity;
    this.shininess=shininess;
    this.metallic=metallic;
    this.fresnel0=fresnel0;
  }
  void operator init(material m) {
    p=copy(m.p);
    opacity=m.opacity;
    shininess=m.shininess;
    metallic=m.metallic;
    fresnel0=m.fresnel0;
  }
  pen diffuse() {return p[0];}
  pen emissive() {return p[1];}
  pen specular() {return p[2];}

  void diffuse(pen q) {p[0]=q;}
  void emissive(pen q) {p[1]=q;}
  void specular(pen q) {p[2]=q;}
}

material operator init() 
{
  return material();
}

void write(file file, string s="", material x, suffix suffix=none)
{
  write(file,s);
  write(file,"{");
  write(file,"diffuse=",x.diffuse());
  write(file,", emissive=",x.emissive());
  write(file,", specular=",x.specular());
  write(file,", opacity=",x.opacity);
  write(file,", shininess=",x.shininess);
  write(file,", metallic=",x.metallic);
  write(file,", F0=",x.fresnel0);
  write(file,"}",suffix);
}

void write(string s="", material x, suffix suffix=endl)
{
  write(stdout,s,x,suffix);
}
  
bool operator == (material m, material n)
{
  return all(m.p == n.p) && m.opacity == n.opacity &&
  m.shininess == n.shininess && m.metallic == n.metallic &&
  m.fresnel0 == n.fresnel0;
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

material emissive(material m, bool colors=false)
{
 return material(black+opacity(m.opacity),colors ? m.emissive() : m.diffuse()+m.emissive(),black,m.opacity,1);
}

pen color(triple normal, material m, light light, transform3 T=light.T) {
  triple[] position=light.position;
  if(invisible((pen) m)) return invisible;
  if(position.length == 0) return m.diffuse();
  normal=unit(transpose(inverse(shiftless(T)))*normal);
  if(settings.twosided) normal *= sgn(normal.z);
  real s=m.shininess*128;
  real[] Diffuse=rgba(m.diffuse());
  real[] Specular=rgba(m.specular());
  real[] p=rgba(m.emissive());
  real[] diffuse={0,0,0,0};
  real[] specular={0,0,0,0};
  for(int i=0; i < position.length; ++i) {
    triple L=position[i];
    real dotproduct=abs(dot(normal,L));
    diffuse += dotproduct*light.diffuse[i];
    dotproduct=abs(dot(normal,unit(L+Z)));
    // Phong-Blinn model of specular reflection
      specular += dotproduct^s*light.specular[i];
  }
  p += diffuse*Diffuse;
  // Apply specularfactor to partially compensate non-pixel-based rendering.
  p += specular*Specular*light.specularfactor;
  return rgb(p[0],p[1],p[2])+opacity(opacity(m.diffuse()));
}

light operator * (transform3 t, light light)
{
  light light=light(light);
  return light;
}

light operator cast(triple v) {return light(v);}

light Viewport=light(specularfactor=3,(0.25,-0.25,1));

light White=light(new pen[] {rgb(0.38,0.38,0.45),rgb(0.6,0.6,0.67),
                             rgb(0.5,0.5,0.57)},specularfactor=3,
  new triple[] {(-2,-1.5,-0.5),(2,1.1,-2.5),(-0.5,0,2)});

light Headlamp=light(gray(0.8),specular=gray(0.7),
                     specularfactor=3,dir(42,48));

currentlight=Headlamp;

light nolight;

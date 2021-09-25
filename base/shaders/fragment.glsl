struct Material
{
  vec4 diffuse,emissive,specular;
  vec4 parameters;
};

struct Light
{
  vec3 direction;
  vec3 color;
};

struct Fragment
{
  vec4 color;
  float depth;
};

uniform int nlights;
uniform Light lights[max(Nlights,1)];

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
in vec3 ViewPosition;
#endif
in vec3 Normal;
vec3 normal;
#endif

#ifdef COLOR
in vec4 Color;
#endif

layout(binding=0) coherent buffer Count {
  uint count[];
};

layout(binding=1) coherent buffer Offset {
  uint offset[];
};

layout(binding=2) coherent buffer list {
  Fragment fragments[];
};

flat in int materialIndex;
out vec4 outColor;
vec4 tempColor;

uniform uint width;

// PBR material parameters
vec3 Diffuse; // Diffuse for nonmetals, reflectance for metals.
vec3 Specular; // Specular tint for nonmetals
float Metallic; // Metallic/Nonmetals parameter
float Fresnel0; // Fresnel at zero for nonmetals
float Roughness2; // roughness squared, for smoothing

#ifdef NORMAL
// h is the halfway vector between normal and light direction
// GGX Trowbridge-Reitz Approximation
float NDF_TRG(vec3 h)
{
  float ndoth=max(dot(normal,h),0.0);
  float alpha2=Roughness2*Roughness2;
  float denom=ndoth*ndoth*(alpha2-1.0)+1.0;
  return denom != 0.0 ? alpha2/(denom*denom) : 0.0;
}

float GGX_Geom(vec3 v)
{
  float ndotv=max(dot(v,normal),0.0);
  float ap=1.0+Roughness2;
  float k=0.125*ap*ap;
  return ndotv/((ndotv*(1.0-k))+k);
}

float Geom(vec3 v, vec3 l)
{
  return GGX_Geom(v)*GGX_Geom(l);
}

// Schlick's approximation
float Fresnel(vec3 h, vec3 v, float fresnel0)
{
  float a=1.0-max(dot(h,v),0.0);
  float b=a*a;
  return fresnel0+(1.0-fresnel0)*b*b*a;
}

vec3 BRDF(vec3 viewDirection, vec3 lightDirection)
{
  vec3 lambertian=Diffuse;
  // Cook-Torrance model
  vec3 h=normalize(lightDirection+viewDirection);

  float omegain=max(dot(viewDirection,normal),0.0);
  float omegaln=max(dot(lightDirection,normal),0.0);

  float D=NDF_TRG(h);
  float G=Geom(viewDirection,lightDirection);
  float F=Fresnel(h,viewDirection,Fresnel0);

  float denom=4.0*omegain*omegaln;
  float rawReflectance=denom > 0.0 ? (D*G)/denom : 0.0;

  vec3 dielectric=mix(lambertian,rawReflectance*Specular,F);
  vec3 metal=rawReflectance*Diffuse;

  return mix(dielectric,metal,Metallic);
}
#endif

void main()
{
  uint headIndex=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  vec4 diffuse;
  vec4 emissive;

  Material m;
#ifdef TRANSPARENT
  m=Materials[abs(materialIndex)-1];
  emissive=m.emissive;
  if(materialIndex >= 0)
    diffuse=m.diffuse;
  else {
    diffuse=Color;
#if Nlights == 0
    emissive += Color;
#endif
  }
#else
  m=Materials[int(materialIndex)];
  emissive=m.emissive;
#ifdef COLOR
  diffuse=Color;
#if Nlights == 0
  emissive += Color;
#endif
#else
  diffuse=m.diffuse;
#endif
#endif

#if defined(NORMAL) && Nlights > 0
  Specular=m.specular.rgb;
  vec4 parameters=m.parameters;
  Roughness2=1.0-parameters[0];
  Roughness2=Roughness2*Roughness2;
  Metallic=parameters[1];
  Fresnel0=parameters[2];
  Diffuse=diffuse.rgb;

  // Given a point x and direction \omega,
  // L_i=\int_{\Omega}f(x,\omega_i,\omega) L(x,\omega_i)(\hat{n}\cdot \omega_i)
  // d\omega_i, where \Omega is the hemisphere covering a point,
  // f is the BRDF function, L is the radiance from a given angle and position.

  normal=normalize(Normal);
  normal=gl_FrontFacing ? normal : -normal;
#ifdef ORTHOGRAPHIC
  vec3 viewDir=vec3(0.0,0.0,1.0);
#else
  vec3 viewDir=-normalize(ViewPosition);
#endif
  // For a finite point light, the rendering equation simplifies.
  vec3 color=emissive.rgb;
  for(int i=0; i < nlights; ++i) {
    Light Li=lights[i];
    vec3 L=Li.direction;
    float cosTheta=max(dot(normal,L),0.0); // $\omega_i \cdot n$ term
    vec3 radiance=cosTheta*Li.color;
    color += BRDF(viewDir,L)*radiance;
  }

  tempColor=vec4(color,diffuse.a);
#else
  tempColor=emissive;
#endif

  uint listIndex=offset[headIndex]+atomicAdd(count[headIndex],1u);
  fragments[listIndex].color=tempColor;
  fragments[listIndex].depth=gl_FragCoord.z;
#ifdef TRANSPARENT
  discard;
#else
  outColor=tempColor;
#endif
}

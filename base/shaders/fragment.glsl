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

flat in int materialIndex;
out vec4 outColor;

// PBR material parameters
vec3 Diffuse; // Diffuse for nonmetals, reflectance for metals.
vec3 Specular; // Specular tint for nonmetals
float Metallic; // Metallic/Nonmetals parameter
float Fresnel0; // Fresnel at zero for nonmetals
float Roughness2; // roughness squared, for smoothing
float Roughness;

#ifdef USE_IBL
uniform sampler2D reflDiffuse;
uniform sampler2D IBLRefl;
uniform sampler3D reflectionMap;
#endif

const float PI=acos(-1.0);
const float twopi=2*PI;
const float halfpi=PI/2;

// (x,y,z) -> (r,theta,phi);
// theta -> [0,\pi]: colatitude
// phi -> [0, 2\pi]: longitude
vec3 cart2sphere(vec3 cart)
{
  float x=cart.x;
  float y=cart.z;
  float z=cart.y;

  float r=length(cart);
  float phi=atan(-y,-x);
  float theta=acos(z/r);

  return vec3(r,phi,theta);
}

vec2 normalizedAngle(vec3 cartVec)
{
  vec3 sphericalVec=cart2sphere(cartVec);
  sphericalVec.y=sphericalVec.y/(2*PI)+PI;
  sphericalVec.z=sphericalVec.z/PI;
  return sphericalVec.yz;
}
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

float sigmoid(float x, float bias, float scale)
{
  return 1/(1+exp(-1*scale*(x-bias)));
}

void main()
{
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
  Roughness=1.0-parameters[0];
  Roughness2=Roughness*Roughness;
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

#ifdef USE_IBL
  // PBR Reflective lights
  vec3 pointLightColor=color;
  //
  // based on the split sum formula approximation
  // L(v)=\int_\Omega L(l)f(l,v) \cos \theta_l
  // which, by the split sum approiximation (assuming independence+GGX distrubition),
  // roughly equals (within a margin of error)
  // [\int_\Omega L(l) ] * [\int_\Omega f(l,v) \cos \theta_l].
  // the first term is the reflectance irradiance integral

  normal=normalize(normal);
  viewDir=normalize(viewDir);
  vec3 reflectVec=normalize(reflect(-viewDir,normal));
  vec3 reflDiffuse=diffuse.rgb*texture2D(reflDiffuse,normalizedAngle(normal)).rgb;

  vec2 reflCoord=normalizedAngle(reflectVec);

  float roughnessSampler=clamp(Roughness,0.005,0.995);
  vec3 reflColor=texture(reflectionMap, vec3(reflCoord, roughnessSampler)).rgb;
  vec2 reflIBL=texture(IBLRefl, vec2(dot(normal, viewDir), roughnessSampler)).rg;

  float specMultiplier=Fresnel0*reflIBL.x+reflIBL.y;

  vec3 dielectricColor=reflDiffuse+(specMultiplier*reflColor);
  vec3 metallicColor=diffuse.rgb*reflColor;
  vec3 finalIBLColor=mix(dielectricColor,metallicColor,Metallic);

  // float test=sigmoid(normal.z,0.85,600);
  outColor=vec4(finalIBLColor+0*color,diffuse.a);
#else
  outColor=vec4(color,diffuse.a);
#endif
#else    
  outColor=emissive;
#endif      
}

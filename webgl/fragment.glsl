#ifdef WEBGL2
#define IN in
out vec4 outValue;
#define OUTVALUE outValue
#else
#define IN varying
#define OUTVALUE gl_FragColor
#endif

#ifdef WEBGL2
flat in int MaterialIndex;

struct Material {
  vec4 diffuse,emissive,specular;
  vec4 parameters;
};

uniform Material Materials[Nmaterials];

vec4 diffuse;
vec3 specular;
float roughness,metallic,fresnel0;
vec4 emissive;

#ifdef COLOR
in vec4 Color;
#endif

#else
IN vec4 diffuse;
IN vec3 specular;
IN float roughness,metallic,fresnel0;
IN vec4 emissive;
#endif

#ifdef NORMAL

#ifndef ORTHOGRAPHIC
IN vec3 ViewPosition;
#endif
IN vec3 Normal;

vec3 normal;

struct Light {
  vec3 direction;
  vec3 color;
};

uniform Light Lights[Nlights];

#ifdef USE_IBL
uniform sampler2D reflBRDFSampler;
uniform sampler2D diffuseSampler;
uniform sampler2D reflImgSampler;

const float pi=acos(-1.0);
const float piInv=1.0/pi;
const float twopi=2.0*pi;
const float twopiInv=1.0/twopi;

// (x,y,z) -> (r,theta,phi);
// theta -> [0,pi]: colatitude
// phi -> [-pi,pi]: longitude
vec3 cart2sphere(vec3 cart)
{
  float x=cart.x;
  float y=cart.z;
  float z=cart.y;

  float r=length(cart);
  float theta=r > 0.0 ? acos(z/r) : 0.0;
  float phi=atan(y,x);

  return vec3(r,theta,phi);
}

vec2 normalizedAngle(vec3 cartVec)
{
  vec3 sphericalVec=cart2sphere(cartVec);
  sphericalVec.y=sphericalVec.y*piInv;
  sphericalVec.z=0.75-sphericalVec.z*twopiInv;
  return sphericalVec.zy;
}

vec3 IBLColor(vec3 viewDir)
{
  vec3 IBLDiffuse=diffuse.rgb*texture(diffuseSampler,normalizedAngle(normal)).rgb;
  vec3 reflectVec=normalize(reflect(-viewDir,normal));
  vec2 reflCoord=normalizedAngle(reflectVec);
  vec3 IBLRefl=textureLod(reflImgSampler,reflCoord,roughness*ROUGHNESS_STEP_COUNT).rgb;
  vec2 IBLbrdf=texture(reflBRDFSampler,vec2(dot(normal,viewDir),roughness)).rg;
  float specularMultiplier=fresnel0*IBLbrdf.x+IBLbrdf.y;
  vec3 dielectric=IBLDiffuse+specularMultiplier*IBLRefl;
  vec3 metal=diffuse.rgb*IBLRefl;
  return mix(dielectric,metal,metallic);
}
#else
float Roughness2;
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

float Fresnel(vec3 h, vec3 v, float fresnel0)
{
  float a=1.0-max(dot(h,v),0.0);
  float b=a*a;
  return fresnel0+(1.0-fresnel0)*b*b*a;
}

// physical based shading using UE4 model.
vec3 BRDF(vec3 viewDirection, vec3 lightDirection)
{
  vec3 lambertian=diffuse.rgb;
  vec3 h=normalize(lightDirection+viewDirection);

  float omegain=max(dot(viewDirection,normal),0.0);
  float omegaln=max(dot(lightDirection,normal),0.0);

  float D=NDF_TRG(h);
  float G=Geom(viewDirection,lightDirection);
  float F=Fresnel(h,viewDirection,fresnel0);

  float denom=4.0*omegain*omegaln;
  float rawReflectance=denom > 0.0 ? (D*G)/denom : 0.0;

  vec3 dielectric=mix(lambertian,rawReflectance*specular,F);
  vec3 metal=rawReflectance*diffuse.rgb;

  return mix(dielectric,metal,metallic);
}
#endif

#endif

void main(void)
{
#ifdef WEBGL2
#ifdef NORMAL
  Material m;
#ifdef TRANSPARENT
  m=Materials[abs(MaterialIndex)-1];
  emissive=m.emissive;
  if(MaterialIndex >= 0)
    diffuse=m.diffuse;
  else {
    diffuse=Color;
#if nlights == 0
    emissive += Color;
#endif
  }
#else
  m=Materials[MaterialIndex];
  emissive=m.emissive;
#ifdef COLOR
  diffuse=Color;
#if nlights == 0
    emissive += Color;
#endif
#else
  diffuse=m.diffuse;
#endif // COLOR
#endif // TRANSPARENT
  specular=m.specular.rgb;
  vec4 parameters=m.parameters;
  roughness=1.0-parameters[0];
  metallic=parameters[1];
  fresnel0=parameters[2];
#else
  emissive=Materials[MaterialIndex].emissive;
#endif // NORMAL
#endif // WEBGL2

#if defined(NORMAL) && nlights > 0
  normal=normalize(Normal);
  normal=gl_FrontFacing ? normal : -normal;
#ifdef ORTHOGRAPHIC
  vec3 viewDir=vec3(0.0,0.0,1.0);
#else
  vec3 viewDir=-normalize(ViewPosition);
#endif

vec3 color;
#ifdef USE_IBL
  color=IBLColor(viewDir);
#else
  Roughness2=roughness*roughness;
  color=emissive.rgb;
  for(int i=0; i < nlights; ++i) {
    Light Li=Lights[i];
    vec3 L=Li.direction;
    float cosTheta=max(dot(normal,L),0.0);
    vec3 radiance=cosTheta*Li.color;
    color += BRDF(viewDir,L)*radiance;
  }
#endif
  OUTVALUE=vec4(color,diffuse.a);
#else
  OUTVALUE=emissive;
#endif
}

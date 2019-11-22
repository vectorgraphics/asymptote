#ifdef NORMAL
#ifndef ORTHOGRAPHIC
varying vec3 ViewPosition;
#endif
varying vec3 Normal;
varying vec4 diffuse;
varying vec3 specular;
varying float roughness,metallic,fresnel0;

float Roughness2;
vec3 normal;

struct Light {
  vec3 direction;
  vec3 color;
};

uniform Light Lights[Nlights];

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
  float omegali=max(dot(lightDirection,normal),0.0);
      
  float D=NDF_TRG(h);
  float G=Geom(viewDirection,lightDirection);
  float F=Fresnel(h,viewDirection,fresnel0);
      
  float denom=4.0*omegain*omegali;
  float rawReflectance=denom > 0.0 ? (D*G)/denom : 0.0;
      
  vec3 dielectric=mix(lambertian,rawReflectance*specular,F);
  vec3 metal=rawReflectance*diffuse.rgb;
      
  return mix(dielectric,metal,metallic);
}
#endif
varying vec4 emissive;
    
void main(void)
{
#if defined(NORMAL) && nlights > 0
  normal=normalize(Normal);
  normal=gl_FrontFacing ? normal : -normal;
#ifdef ORTHOGRAPHIC
  vec3 viewDir=vec3(0.0,0.0,1.0);
#else
  vec3 viewDir=-normalize(ViewPosition);
#endif
  Roughness2=roughness*roughness;
  vec3 color=emissive.rgb;
  for(int i=0; i < nlights; ++i) {
    Light Li=Lights[i];
    vec3 L=Li.direction;
    float cosTheta=max(dot(normal,L),0.0);
    vec3 radiance=cosTheta*Li.color;
    color += BRDF(viewDir,L)*radiance;
  }
  gl_FragColor=vec4(color,diffuse.a);
#else
  gl_FragColor=emissive;
#endif
}

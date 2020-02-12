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

#ifdef ENABLE_TEXTURE
uniform sampler2D environmentMap;
const float PI=acos(-1.0);
const float twopi=2*PI;
const float halfpi=PI/2;

const int numSamples=7;

// (x,y,z) -> (r,theta,phi);
// theta -> [0,\pi]: colatitude
// phi -> [0, 2\pi]: longitude
vec3 cart2sphere(vec3 cart)
{
  float x=cart.z;
  float y=cart.x;
  float z=cart.y;

  float r=length(cart);
  float phi=atan(y,x);
  float theta=acos(z/r);

  return vec3(r,phi,theta);
}

vec2 normalizedAngle(vec3 cartVec)
{
  vec3 sphericalVec=cart2sphere(cartVec);
  sphericalVec.y=sphericalVec.y/(2*PI)-0.25;
  sphericalVec.z=sphericalVec.z/PI;
  return sphericalVec.yz;
}
#endif

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

#if defined(ENABLE_TEXTURE) && !defined(COLOR)
  // Experimental environment radiance using Riemann sums;
  // can also do importance sampling.
  vec3 envRadiance=vec3(0.0,0.0,0.0);

  vec3 normalPerp=vec3(-normal.y,normal.x,0.0);
  if(length(normalPerp) == 0.0)
    normalPerp=vec3(1.0,0.0,0.0);

  // we now have a normal basis;
  normalPerp=normalize(normalPerp);
  vec3 normalPerp2=normalize(cross(normal,normalPerp));

  const float step=1.0/numSamples;
  const float phistep=twopi*step;
  const float thetastep=halfpi*step;
  for (int iphi=0; iphi < numSamples; ++iphi) {
    float phi=iphi*phistep;
    for (int itheta=0; itheta < numSamples; ++itheta) {
      float theta=itheta*thetastep;

      vec3 azimuth=cos(phi)*normalPerp+sin(phi)*normalPerp2;
      vec3 L=sin(theta)*azimuth+cos(theta)*normal;

      vec3 rawRadiance=texture(environmentMap,normalizedAngle(L)).rgb;
      vec3 surfRefl=BRDF(Z,L);
      envRadiance += surfRefl*rawRadiance*sin(2.0*theta);
    }
  }
  envRadiance *= halfpi*step*step;
  color += envRadiance.rgb;
#endif
  outColor=vec4(color,diffuse.a);
#else    
  outColor=emissive;
#endif      
}

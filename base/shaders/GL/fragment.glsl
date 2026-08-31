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

uniform uint nlights;
uniform Light lights[max(Nlights,1)];

// sRGB (perceptual) output; set at runtime, so no recompilation is needed
// when the setting changes (true = convert, false = output linear)
uniform bool srgb;

// IBL shading; set at runtime, so enabling/disabling IBL needs no shader
// recompilation. The IBL samplers are always declared; they are only
// sampled when this flag is set.
uniform bool ibl;

const float gamma=2.2;
const float invGamma=1.0/gamma;

/**
 * @brief Converts linear color (measuring photon count) to srgb (what our brain thinks
 * is the brightness
 * example linearToPerceptual(vec3(0.5)) is approximately vec3(0.729)
 */
vec3 linearToPerceptual(vec3 inColor)
{
  // an actual 0.5 brightness (half amount of photons) would
  // look brighter than what our eyes think is "half" light
  return pow(inColor, vec3(invGamma));
}

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

flat in int materialIndex;
out vec4 outColor;

#ifdef NORMAL
// PBR material parameters (received from vertex shader)
in vec4 diffuse;
in vec3 specular;
in float Roughness2_in,Roughness_in,Metallic_in,Fresnel0_in;
in vec4 emissive;
#endif

// PBR material parameters (used by BRDF functions)
vec3 Diffuse;
vec3 Specular;
float Metallic;
float Fresnel0;
float Roughness2;
float Roughness;

#ifdef HAVE_SSBO

layout(binding=0, std430) buffer offsetBuffer
{
  uint maxDepth;
  uint offset[];
};

#ifndef GPUINDEXING
layout(binding=2, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};
#endif

layout(binding=4, std430) buffer fragmentBuffer
{
  vec4 fragment[];
};

layout(binding=5, std430) buffer depthBuffer
{
  float depth[];
};

layout(binding=6, std430) buffer opaqueBuffer
{
  vec4 opaqueColor[];
};

layout(binding=7, std430) buffer opaqueDepthBuffer
{
  float opaqueDepth[];
};

#ifdef GPUCOMPRESS
layout(binding=1, std430) buffer indexBuffer
{
  uint index[];
};
#define INDEX(pixel) index[pixel]
#else
#define INDEX(pixel) pixel
#endif

uniform uint width;

#endif

#ifdef NORMAL

in vec3 ViewPosition;
in vec3 Normal;
vec3 normal;

// IBL samplers are always declared; they are only sampled when the `ibl`
// uniform is set
uniform sampler2D reflBRDFSampler;
uniform sampler2D diffuseSampler;
uniform sampler3D reflImgSampler;

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
  //
  // based on the split sum formula approximation
  // L(v)=\int_\Omega L(l)f(l,v) \cos \theta_l
  // which, by the split sum approiximation (assuming independence+GGX distrubition),
  // roughly equals (within a margin of error)
  // [\int_\Omega L(l)] * [\int_\Omega f(l,v) \cos \theta_l].
  // the first term is the reflectance irradiance integral

  vec3 IBLDiffuse=Diffuse*texture(diffuseSampler,normalizedAngle(normal)).rgb;
  vec3 reflectVec=normalize(reflect(-viewDir,normal));
  vec2 reflCoord=normalizedAngle(reflectVec);
  vec3 IBLRefl=texture(reflImgSampler,vec3(reflCoord,Roughness)).rgb;
  vec2 IBLbrdf=texture(reflBRDFSampler,vec2(dot(normal,viewDir),Roughness)).rg;
  float specularMultiplier=Fresnel0*IBLbrdf.x+IBLbrdf.y;
  vec3 dielectric=IBLDiffuse+specularMultiplier*IBLRefl;
  vec3 metal=Diffuse*IBLRefl;
  return mix(dielectric,metal,Metallic);
}

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
#if defined(NORMAL) && Nlights > 0
  Diffuse=diffuse.rgb;
  Specular=specular.rgb;
  Roughness=Roughness_in;
  Roughness2=Roughness2_in;
  Metallic=Metallic_in;
  Fresnel0=Fresnel0_in;

  // Given a point x and direction \omega,
  // L_i=\int_{\Omega}f(x,\omega_i,\omega) L(x,\omega_i)(\hat{n}\cdot \omega_i)
  // d\omega_i, where \Omega is the hemisphere covering a point,
  // f is the BRDF function, L is the radiance from a given angle and position.

  normal=normalize(Normal);
  normal=gl_FrontFacing ? normal : -normal;
  // Orthographic mode is handled by the vertex shader, which writes
  // ViewPosition=(0,0,-1); -normalize(ViewPosition) is then (0,0,1)
  vec3 viewDir=-normalize(ViewPosition);
  vec3 color;
  // The IBL flag is a runtime uniform, so enabling/disabling IBL needs no
  // shader recompilation
  if(ibl) {
    color=IBLColor(viewDir);
  } else {
    // For a finite point light, the rendering equation simplifies.
    color=emissive.rgb;
    for(uint i=0u; i < nlights; ++i) {
      Light Li=lights[i];
      vec3 L=Li.direction;
      float cosTheta=max(dot(normal,L),0.0); // $\omega_i \cdot n$ term
      vec3 radiance=cosTheta*Li.color;
      color += BRDF(viewDir,L)*radiance;
    }
  }
  outColor=vec4(color,diffuse.a);
#else
#ifdef NORMAL
  outColor=emissive;
#else
  Material m=Materials[materialIndex];
  outColor=m.emissive;
#endif
#endif

  // All of our calculations have been linear (i.e. by measuring photon counts);
  // if sRGB output is requested, convert to perceptual here. The transparency
  // buffers store linear values; the blend pass applies the same conversion to
  // its output. The framebuffer's sRGB capability is not used.
  vec4 linearColor=outColor;
  if(srgb) {
    vec3 outColorInPerceptualSpace=linearToPerceptual(linearColor.rgb);
    outColor=vec4(outColorInPerceptualSpace,linearColor.a);
  }

#ifndef WIDTH
#ifdef HAVE_SSBO
  uint pixel=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
#if defined(TRANSPARENT) || (!defined(HAVE_INTERLOCK) && !defined(OPAQUE))
  uint element=INDEX(pixel);
#ifdef GPUINDEXING
  uint listIndex=atomicAdd(offset[element],-1u)-1u;
#else
  uint listIndex=offset[element]-atomicAdd(count[element],1u)-1u;
#endif
  fragment[listIndex]=linearColor;
  depth[listIndex]=gl_FragCoord.z;
  // The fragment's color reaches the framebuffer through the blend pass;
  // discarding here matches the Vulkan shader's behavior in all modes
  discard;
#else
#if defined(HAVE_INTERLOCK) && !defined(OPAQUE)
  beginInvocationInterlockARB();
  if(opaqueDepth[pixel] == 0.0 || gl_FragCoord.z < opaqueDepth[pixel])
    {
    opaqueDepth[pixel]=gl_FragCoord.z;
    opaqueColor[pixel]=linearColor;
  }
  endInvocationInterlockARB();
#endif
#endif
#endif
#endif
}

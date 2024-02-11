struct Material
{
  vec4 diffuse, emissive, specular;
  vec4 parameters;
};

struct Light
{
  vec4 direction;
  vec4 color;
};

layout(binding = 0) uniform UniformBufferObject
{
  mat4 projViewMat;
  mat4 viewMat;
  mat4 normMat;
} ubo;

layout(binding = 1, std430) buffer MaterialBuffer
{
  Material materials[];
};

layout(binding = 2, std430) buffer LightBuffer
{
  Light lights[];
};

layout(binding = 3, std430) buffer CountBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding = 4, std430) buffer OffsetBuffer
{
  uint maxDepth;
  uint offset[];
};

layout(binding = 5, std430) buffer FragmentBuffer
{
  vec4 fragment[];
};

layout(binding = 6, std430) buffer DepthBuffer
{
  float depth[];
};

layout(binding = 7, std430) buffer OpaqueBuffer
{
  vec4 opaqueColor[];
};

layout(binding = 8, std430) buffer OpaqueDepthBuffer
{
  float opaqueDepth[];
};

#ifdef GPUCOMPRESS
layout(binding=9, std430) buffer indexBuffer
{
  uint index[];
};
#define INDEX(pixel) index[pixel]
#else
#define INDEX(pixel) pixel
#endif

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 viewPos;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec4 inColor;
layout(location = 4) flat in int materialIndex;

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
  vec4 background;
  // constants[0] = nlights
  // constants[1] = width;
} push;

layout(location = 0) out vec4 outColor;

vec3 Emissive;
vec3 Diffuse;
vec3 Specular;
float Metallic;
float Fresnel0;
float Roughness2;
float Roughness;

vec3 normal;

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

#ifdef USE_IBL

layout(binding=11) uniform sampler2D diffuseSampler;
layout(binding=12) uniform sampler2D reflBRDFSampler;
layout(binding=13) uniform sampler3D reflImgSampler;

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

#else

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
  return GGX_Geom(v) * GGX_Geom(l);
}

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

void main() {

  uint nlights = push.constants[0];

  Material mat;

#ifdef GENERAL
  mat = materials[abs(materialIndex) - 1];

  if (materialIndex < 0) {
    mat.diffuse = inColor;
#ifdef NOLIGHTS
      mat.emissive += inColor;
#endif /*NOLIGHTS*/
  }

#else

  mat = materials[materialIndex];

#ifdef COLOR
  mat.diffuse = inColor;
#endif /*COLOR*/
#endif /*GENERAL*/

  outColor = mat.emissive;

#ifdef NORMAL

  Diffuse = mat.diffuse.rgb;
  Specular = mat.specular.rgb;
  Roughness = 1.f - mat.parameters[0];
  Metallic = mat.parameters[1];
  Fresnel0 = mat.parameters[2];
  Roughness2 = Roughness * Roughness;

#ifdef ORTHOGRAPHIC
  vec3 viewDirection=vec3(0.0,0.0,1.0);
#else
  vec3 viewDirection=-normalize(viewPos);
#endif
  normal = normalize(norm);

  if (!gl_FrontFacing)
      normal = -normal;

#ifdef USE_IBL
  outColor=vec4(IBLColor(viewDirection), outColor.a);
#else
  for (int i = 0; i < nlights; i++)
  {
      Light light = lights[i];

      vec3 radiance = max(dot(normal, light.direction.xyz), 0.0) * light.color.rgb;
      outColor += vec4(BRDF(viewDirection, light.direction.xyz) * radiance, 0.0);
  }

  outColor = vec4(outColor.rgb, mat.diffuse.a);
#endif /*USE_IBL*/
#endif /*NORMAL*/

  // for reasons, the swapchain/FXAA shader expects a "perceptual" color,
  // while all of our calculations have been linear (i.e. by measuring photon counts)
  // (e.g. our 0.5 is much much brighter than what swap chain/monitor thinks 0.5 is)
  // need to give the output image the color our brain perceives with the same photon count
  // as the original pixel
  vec3 outColorInPerceptualSpace=linearToPerceptual(outColor.rgb);
  outColor=vec4(outColorInPerceptualSpace,outColor.a);

#ifndef WIDTH // TODO DO NOT DO THE DEPTH COMPARISON WHEN NO TRANSPARENT OBJECTS!
  uint pixel=uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x);
#if defined(TRANSPARENT) || (!defined(HAVE_INTERLOCK) && !defined(OPAQUE))
  uint element=INDEX(pixel);
#ifdef GPUINDEXING
  uint listIndex=atomicAdd(offset[element],-1u)-1u;
#else
  uint listIndex=offset[element]-atomicAdd(count[element],1u)-1u;
#endif /*GPUINDEXING*/
  fragment[listIndex]=outColor;
  depth[listIndex]=gl_FragCoord.z;
#ifndef WIREFRAME
  discard;
#endif /*WIREFRAME*/
#else
#if defined(HAVE_INTERLOCK) && !defined(OPAQUE)
  beginInvocationInterlockARB();
  if(opaqueDepth[pixel] == 0.0 || gl_FragCoord.z < opaqueDepth[pixel])
    {
    opaqueDepth[pixel]=gl_FragCoord.z;
    opaqueColor[pixel]=outColor;
  }
  endInvocationInterlockARB();
#endif
#endif
#endif
}

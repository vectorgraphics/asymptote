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

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

flat in int materialIndex;
out vec4 outColor;
// PBR material parameters
vec3 Diffuse; // Diffuse for nonmetals, reflectance for metals.
vec3 Specular; // Specular tint for nonmetals
float Metallic; // Metallic/Nonmetals parameter
float Fresnel0; // Fresnel at zero for nonmetals
float Roughness2; // roughness squared, for smoothing
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

/*

// clipIndex clip[]={clipIndex(0,36),clipIndex(0,36),clipIndex(36,12)};
// surface A 0,1 (cube): clip[0]
// surface B 1,2 (cube,tetrahedron): clip[1],clip[2]

*/

struct clipIndex {
  int offset,size;
};

layout(binding=9, std430) buffer clipVertexBuffer {
vec4 vertex[];
};

layout(binding=10, std430) buffer clipBuffer {
int clipindex[];
};

layout(binding=11, std430) buffer clipIndexBuffer {
  clipIndex clip[];
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

#ifndef ORTHOGRAPHIC
in vec3 ViewPosition;
#endif
in vec3 Normal;
vec3 normal;

#ifdef USE_IBL
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
#else
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

#endif

#ifdef COLOR
in vec4 Color;
#endif

float orient(vec3 a, vec3 b, vec3 c, vec3 d) {
  return dot(cross(a-d,b-d), c-d);
}

void checkCoplanar(vec3 vertex1, vec3 vertex2, vec3 testPoint, float Epsilon, out bool check, inout vec3 outside) {
  vec3 normal=normalize(cross(testPoint-vertex1,vertex2-vertex1));
  vec3 H=testPoint+normal;
  if (orient(vertex1,testPoint,vertex2,H) != 0 && orient(vertex1,testPoint,vertex2,outside) == 0) {
    outside += normal*Epsilon;
    check = true;
  }
}

#define Vertex(i) vertex[clipindex[i]]

vec3 nonCoplanarOutsidePoint(vec3 v, uint startIndex, uint endIndex) {
  uint n = vertex.length();
  vec3 m = Vertex(0).xyz;
  for (uint i=0;i<n;++i) m = min(m,Vertex(i).xyz);
  vec3 M = Vertex(0).xyz;
  for (uint i=0;i<n;++i) M = max(M,Vertex(i).xyz);

  vec3 outside=10*M-m; // TODO: Revert to 2*M-m;
  float norm=length(M-m);
  float Epsilon=norm*FLT_EPSILON;

  // check that the outside point is not coplanar with any of the faces of the
  // of the polyhedron. if it is, move it a little bit in the direction of the
  // normal and recheck all the faces
  bool check=true;
  while (check) {
    check = false;
    // check each face
    for (uint i=startIndex;i<endIndex; i += 3) {
      vec3 a=Vertex(i).xyz;
      vec3 b=Vertex(i+1).xyz;
      vec3 c=Vertex(i+2).xyz;
      // for each face (3 vertices), check each edge
      checkCoplanar(a,b,v,Epsilon,check,outside);
      checkCoplanar(b,c,v,Epsilon,check,outside);
      checkCoplanar(c,a,v,Epsilon,check,outside);
    }
  }
  return outside;
}

void discardIfInsideFace(vec3 v, vec3 a, vec3 b, vec3 c) {
  // v is test point, a,b,c are vertices of the face
  vec3 m=min(min(a,b),c);
  vec3 M=max(max(a,b),c);

  vec3 outside=2*M-m;
  float norm=length(M-m);
  float Epsilon=norm*FLT_EPSILON;

  vec3 n=normalize(cross(c-a,b-a));
  vec3 normal=norm*n;
  vec3 H=v+normal;

  // project the outside point on to the plane defined by the face
  outside -= dot(outside,n)*n;

  vec3 currentFace[3]=vec3[3](a,b,c);  // put in array for iteration
  // make sure the outside point is not colinear with any of the edges of the
  // face
  bool check=true;
  while(check) {
    check=false;
    for(uint i=0;i<3;++i) {
      vec3 u=currentFace[i];
      if (u == v) discard;  // test point is a vertex
      if (orient(u,v,outside,H) == 0) {
        vec3 normal=normalize(cross(v-u,H-u));
        outside += normal*Epsilon;
        outside -= dot(outside,n)*n;
        check=true;
      }
    }
  }


  int count=0;
  vec3 z0=currentFace[2];
  vec3 z=v;
  for(uint i=0; i<3; ++i) {
    vec3 z1=currentFace[i];
    float s1=sign(orient(z,z0,z1,H));
    if (s1 == 0) {
      // insidesegment in 3d
      if (z == z1 || z == z0) discard;
      if (z0 == z1) continue;
      vec3 h=cross(z1-z0,normal);
      float s1_=sign(orient(z0,z,h,H));
      float s2_=sign(orient(z1,z,h,H));
      if (s1_ != s2_) {
        discard;
      }
      continue;
    }

    float s2=sign(orient(outside,z0,z1,H));

    if (s1 == s2) {
      continue;
    }

    float s3=sign(orient(z,outside,z0,H));
    float s4=sign(orient(z,outside,z1,H));
    if (s3 != s4) {
      count += int(s3);
    }

    z0=z1;
  }

  if (count != 0) discard;
}

void discardIfInside(vec3 v, uint startIndex, uint endIndex) {
  vec3 outside=nonCoplanarOutsidePoint(v, startIndex, endIndex);
  int count=0;
  for (uint i=startIndex;i<endIndex; i += 3) {
    vec3 a=Vertex(i).xyz;
    vec3 b=Vertex(i+1).xyz;
    vec3 c=Vertex(i+2).xyz;

    float s1=sign(orient(v,a,b,c));
    if (s1 == 0) {
      // s1 == 0 is the case where the test point lies on the planar extension
      // of the face. check if it lies within the face
      discardIfInsideFace(v,a,b,c);
      continue;
    }

    float s2=sign(orient(outside,a,b,c));
    if (s1 == s2) {
      // s1 == s2 means that the test point has the same sidedness as the
      // outside point for this face, indicating that it is also outside and
      // has no winding number contribution
      continue;
    }

    float s3=sign(orient(v,outside,a,b));
    float s4=sign(orient(v,outside,b,c));
    float s5=sign(orient(v,outside,c,a));

    if (s3 == s4 && s4 == s5) {
      count += int(s3);
    }
  }

  if (count != 0) discard;
}

void doClipping(vec3 v) {
  uint clipLength=clip.length();
  for (uint i=0; i < clipLength; ++i) {
    clipIndex c=clip[i];
    discardIfInside(v, c.offset, c.offset+c.size);
  }
}

in vec4 V;

void main()
{
  vec3 v=V.xyz/V.w;
  doClipping(v);

  vec4 diffuse;
  vec4 emissive;

  Material m;
#ifdef GENERAL
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
  m=Materials[materialIndex];
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
  vec3 color;
#ifdef USE_IBL
  color=IBLColor(viewDir);
#else
  // For a finite point light, the rendering equation simplifies.
  color=emissive.rgb;
  for(uint i=0u; i < nlights; ++i) {
    Light Li=lights[i];
    vec3 L=Li.direction;
    float cosTheta=max(dot(normal,L),0.0); // $\omega_i \cdot n$ term
    vec3 radiance=cosTheta*Li.color;
    color += BRDF(viewDir,L)*radiance;
  }
#endif
  outColor=vec4(color,diffuse.a);
#else
  outColor=emissive;
#endif

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
  fragment[listIndex]=outColor;
  depth[listIndex]=gl_FragCoord.z;
#ifndef WIREFRAME
  discard;
#endif
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
#endif
}

attribute vec3 position;
#ifdef WIDTH
attribute float width;
#endif
#ifdef NORMAL
attribute vec3 normal;
#endif
attribute float materialIndex;
#ifdef COLOR
attribute vec4 color;
#endif

uniform mat3 normMat;
uniform mat4 viewMat;
uniform mat4 projViewMat;

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
varying vec3 ViewPosition;
#endif
varying vec3 Normal;
#endif
varying vec4 diffuse;
varying vec3 specular;
varying float roughness,metallic,fresnel0;
varying vec4 emissive;

struct Material {
  vec4 diffuse,emissive,specular;
  float shininess,metallic,fresnel0;
};

uniform Material Materials[Nmaterials];

void main(void)
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  ViewPosition=(viewMat*v).xyz;
#endif      
  Normal=normal*normMat;
        
  Material m;
#ifdef TRANSPARENT
  m=Materials[int(abs(materialIndex))-1];
  if(materialIndex >= 0.0) {
    diffuse=m.diffuse;
    emissive=m.emissive;
  } else {
    diffuse=color;
#if nlights > 0
    emissive=vec4(0.0);
#else
    emissive=color;
#endif
  }
#else
  m=Materials[int(materialIndex)];
#ifdef COLOR
  diffuse=color;
#if nlights > 0
  emissive=vec4(0.0);
#else
  emissive=color;
#endif
#else
  diffuse=m.diffuse;
  emissive=m.emissive;
#endif
#endif
  specular=m.specular.rgb;
  roughness=1.0-m.shininess;
  metallic=m.metallic;
  fresnel0=m.fresnel0;
#else
  emissive=Materials[int(materialIndex)].emissive;
#endif
#ifdef WIDTH
  gl_PointSize=width;
#endif
}
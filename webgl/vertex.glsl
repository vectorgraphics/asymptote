#ifdef WEBGL2
#define IN in
#define OUT out
#else
#define IN attribute
#define OUT varying
#endif

IN vec3 position;
#ifdef WIDTH
IN float width;
#endif
#ifdef NORMAL
IN vec3 normal;
#endif

IN float materialIndex;

#ifdef WEBGL2
flat out int MaterialIndex;
#ifdef COLOR
OUT vec4 Color;
#endif

#else
OUT vec4 diffuse;
OUT vec3 specular;
OUT float roughness,metallic,fresnel0;
OUT vec4 emissive;

struct Material {
  vec4 diffuse,emissive,specular;
  vec4 parameters;
};

uniform Material Materials[Nmaterials];
#endif

#ifdef COLOR
IN vec4 color;
#endif

uniform mat3 normMat;
uniform mat4 viewMat;
uniform mat4 projViewMat;

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
OUT vec3 ViewPosition;
#endif
OUT vec3 Normal;
#endif

void main(void)
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  ViewPosition=(viewMat*v).xyz;
#endif
  Normal=normalize(normal*normMat);
#endif

#ifdef WEBGL2
  MaterialIndex=int(materialIndex);
#ifdef COLOR
  Color=color;
#endif
#else
#ifdef NORMAL
  Material m;
#ifdef TRANSPARENT
  m=Materials[int(abs(materialIndex))-1];
  emissive=m.emissive;
  if(materialIndex >= 0.0)
    diffuse=m.diffuse;
  else {
    diffuse=color;
#if nlights == 0
    emissive += color;
#endif
  }
#else
  m=Materials[int(materialIndex)];
  emissive=m.emissive;
#ifdef COLOR
  diffuse=color;
#if nlights == 0
    emissive += color;
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
  emissive=Materials[int(materialIndex)].emissive;
#endif // NORMAL
#endif // WEBGL2

#ifdef WIDTH
  gl_PointSize=width;
#endif
}

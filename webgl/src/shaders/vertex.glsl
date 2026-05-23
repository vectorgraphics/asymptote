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
OUT float roughness,metallic,fresnel0,lightOn;
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

uniform bool orthographic;
uniform mat3 normMat;
uniform mat4 viewMat;
uniform mat4 projViewMat;

#ifdef NORMAL
OUT vec3 ViewPosition;
OUT vec3 Normal;
#endif

void main(void)
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;

#ifdef NORMAL
  ViewPosition=orthographic ? vec3(0.0,0.0,-1.0) : (viewMat*v).xyz;
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
#ifdef GENERAL
  m=Materials[int(abs(materialIndex))-1];
  emissive=m.emissive;
  if(materialIndex >= 0.0)
    diffuse=m.diffuse;
  else {
    if (m.parameters[3] != 0.0) {
      diffuse=color;
#if nlights == 0
      emissive += color;
#endif
    } else {
      emissive += color;
      diffuse = m.diffuse;
    }
  }
#else
  m=Materials[int(materialIndex)];
  emissive=m.emissive;
#ifdef COLOR
  if (m.parameters[3] != 0.0) {
    diffuse=color;
#if nlights == 0
      emissive += color;
#endif
  } else {
    emissive += color;
    diffuse = m.diffuse;
  }
#else
  diffuse=m.diffuse;
#endif // COLOR
#endif // GENERAL
  specular=m.specular.rgb;
  vec4 parameters=m.parameters;
  roughness=1.0-parameters[0];
  metallic=parameters[1];
  fresnel0=parameters[2];
  lightOn=parameters[3];
#else
  emissive=Materials[int(materialIndex)].emissive;
#endif // NORMAL
#endif // WEBGL2

#ifdef WIDTH
  gl_PointSize=width;
#endif
}

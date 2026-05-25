in vec3 position;

#ifdef NORMAL

#ifndef ORTHOGRAPHIC
uniform mat4 viewMat;
out vec3 ViewPosition;
#endif

uniform mat3 normMat;
in vec3 normal;
out vec3 Normal;

#endif

#ifdef MATERIAL
in int material;
flat out int materialIndex;
#endif

#ifdef COLOR
in vec4 color;
#endif

#ifdef WIDTH
in float width;
#endif

uniform mat4 projViewMat;

#ifdef NORMAL
struct Material
{
  vec4 diffuse,emissive,specular;
  vec4 parameters;
};

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

out vec4 diffuse;
out vec3 specular;
out float Roughness2_in,Roughness_in,Metallic_in,Fresnel0_in;
out vec4 emissive;
#endif

void main()
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  ViewPosition=(viewMat*v).xyz;
#endif
  Normal=normalize(normal*normMat);

  Material m;
#ifdef GENERAL
  materialIndex=material;
  m=Materials[abs(material)-1];
  emissive=m.emissive;
  if(material >= 0)
    diffuse=m.diffuse;
  else {
    if (m.parameters[3] != 0) {
      diffuse=color;
#if Nlights == 0
      emissive += color;
#endif
    } else {
      emissive += color;
      diffuse = m.diffuse;
    }
  }
#else
  materialIndex=material;
  m=Materials[material];
  emissive=m.emissive;
#ifdef COLOR
  if (m.parameters[3] != 0) {
    diffuse=color;
#if Nlights == 0
    emissive += color;
#endif
  } else {
    emissive += color;
    diffuse = m.diffuse;
  }
#else
  diffuse=m.diffuse;
#endif
#endif
  specular=m.specular.rgb;
  vec4 parameters=m.parameters;
  Roughness_in=1.0-parameters[0];
  Roughness2_in=Roughness_in*Roughness_in;
  Metallic_in=parameters[1];
  Fresnel0_in=parameters[2];
#else
#ifdef MATERIAL
  materialIndex=material;
#endif
#endif

#ifdef WIDTH
  gl_PointSize=width;
#endif
}

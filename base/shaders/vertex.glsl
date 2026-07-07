layout(binding=0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat3 normMat;
} ubo;

layout(location=0) in vec3 inPosition;
#ifdef NORMAL
layout(location=1) in vec3 inNormal;
layout(location=1) out vec3 viewPosition;
layout(location=2) out vec3 normal;
#endif
#ifdef MATERIAL
layout(location=2) in int inMaterial;
layout(location=4) flat out int materialIndex;
#endif
#ifdef COLOR
layout(location=3) in vec4 inColor;
#endif
#ifdef WIDTH
layout(location=4) in float inWidth;
#endif

layout(location=0) out vec3 position;

#ifdef NORMAL
layout(location=5) out vec4 diffuse;
layout(location=6) out vec3 specular;
layout(location=7) out vec3 params; // roughness, metallic, fresnel0, lightOn
layout(location=8) out vec4 emissive;

struct Material
{
  vec4 diffuse, emissive, specular;
  vec4 parameters;
};

layout(binding = 1, std430) buffer MaterialBuffer
{
  Material materials[];
};
#endif

void main()
{
  vec4 v=vec4(inPosition,1.0);
  gl_Position=ubo.projViewMat*v;
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  viewPosition=(ubo.viewMat*v).xyz;
#endif
  normal=normalize(inNormal*ubo.normMat);

  Material mat;
#ifdef GENERAL
  materialIndex=inMaterial;
  mat=materials[abs(inMaterial)-1];
  emissive=mat.emissive;
  if(inMaterial >= 0)
    diffuse=mat.diffuse;
  else {
    if (mat.parameters[3] != 0) {
      diffuse=inColor;
#ifdef NOLIGHTS
      emissive += inColor;
#endif
    } else {
      emissive += inColor;
      diffuse = mat.diffuse;
    }
  }
#else
  materialIndex=inMaterial;
  mat=materials[inMaterial];
  emissive=mat.emissive;
#ifdef COLOR
  if (mat.parameters[3] != 0) {
    diffuse=inColor;
#ifdef NOLIGHTS
    emissive += inColor;
#endif
  } else {
    emissive += inColor;
    diffuse = mat.diffuse;
  }
#else
  diffuse=mat.diffuse;
#endif
#endif
  specular=mat.specular.rgb;
  params=vec3(1.0-mat.parameters[0], mat.parameters[1], mat.parameters[2]);
#else
#ifdef MATERIAL
  materialIndex=inMaterial;
#endif
#endif

#ifdef WIDTH
  gl_PointSize=inWidth;
#endif
}

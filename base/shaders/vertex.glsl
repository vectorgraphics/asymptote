layout(binding=0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat4 normMat;
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
layout(location=3) out vec4 color;
#endif
#ifdef WIDTH
layout(location=4) in float inWidth;
#endif

layout(location=0) out vec3 position;

void main()
{
  vec4 v=vec4(inPosition,1.0);
  gl_Position=ubo.projViewMat*v;
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  viewPosition=(ubo.viewMat*v).xyz;
#endif
  normal=normalize((vec4(inNormal, 1.0)*ubo.normMat).xyz);
#endif

#ifdef MATERIAL
  materialIndex=inMaterial;
#endif
#ifdef COLOR
  color=inColor;
#endif

#ifdef WIDTH
  gl_PointSize=inWidth;
#endif
}

#version 450
#define MATERIAL
#define COLOR
#define NORMAL
#define TRANSPARENT
#define GENERAL
#define GPUCOMPRESS

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat4 normMat;
} ubo;

layout(location = 0) in vec3 inPosition;
#ifdef NORMAL
layout(location = 1) in vec3 inNormal;
#endif
#ifdef MATERIAL
layout(location = 2) in int inMaterial;
#endif
#ifdef COLOR
layout(location = 3) in vec4 inColor;
#endif
#ifdef WIDTH
layout(location = 4) in float inWidth;
#endif

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 viewPos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec4 color;
layout(location = 4) flat out int materialIndex;

void main() {
    position        = inPosition;
    viewPos         = (ubo.viewMat * vec4(inPosition, 1.0)).xyz;
#ifdef NORMAL
    normal          = normalize((vec4(inNormal, 1.0) * ubo.normMat).xyz);
#endif
#ifdef MATERIAL
    materialIndex   = inMaterial;
#endif
#ifdef COLOR
    color           = inColor;
#endif

    gl_Position     = ubo.projViewMat * vec4(inPosition, 1.0);
#ifdef WIDTH
    gl_PointSize    = inWidth;
#endif
}

#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat4 normMat;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inWidth;
layout(location = 2) in int inMaterial;

layout(location = 0) flat out int materialIndex;

void main() {

    materialIndex   = inMaterial;

    gl_Position     = ubo.projViewMat * vec4(inPosition, 1.0);
    gl_PointSize    = inWidth;
}

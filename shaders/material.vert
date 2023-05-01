#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    vec3 lightPos;
    vec3 viewPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in int inMaterial;

layout(location = 0) out vec3 position;
layout(location = 1) out vec3 normal;
layout(location = 2) flat out int materialIndex;

void main() {
    position = inPosition;
    normal = inNormal;
    materialIndex = inMaterial;
    gl_Position = ubo.projViewMat * vec4(inPosition, 1.0);
}

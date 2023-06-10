#version 450

#define PUSHFLAGS_NOLIGHT (1 << 0)
#define PUSHFLAGS_COLORED (1 << 1)

struct Material
{
    vec4 diffuse, emissive, specular;
    vec4 parameters;
};

struct Light
{
    vec4 direction;
    vec4 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat4 normMat;
} ubo;

layout(binding = 1, std430) buffer MaterialBuffer {
    Material materials[];
};

layout(binding = 2, std430) buffer LightBuffer {
    Light lights[];
};

layout(location = 0) flat in int materialIndex;

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
    // constants[0] = flags
    // constants[1] = nlights
} push;

layout(location = 0) out vec4 outColor;

void main() {

    outColor = materials[materialIndex].emissive;
}

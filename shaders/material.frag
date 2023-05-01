#version 450

struct Material
{
    vec3 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    vec3 lightPos;
    vec3 viewPos;
} ubo;

// layout(binding = 1) buffer MaterialSSBO { Material materials[]; };

// will this data be bound to 0 and cause a problem?
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) flat in int materialIndex;

layout(location = 0) out vec4 outColor;

// could we normalize this stuff in the vertex shader or even in asymptote?
void main() {
    // Material material = materials[materialIndex];
    Material material = Material(vec3(0.0, 0.0, 1.0));
    // ambient
    vec3 ambient = 0.05 * material.color;
    // diffuse
    vec3 lightDir = normalize(ubo.lightPos - position);
    vec3 normal_ = normalize(normal);
    float diff = max(dot(lightDir, normal_), 0.0);
    vec3 diffuse = diff * material.color;
    // specular
    vec3 viewDir = normalize(ubo.viewPos - position);
    vec3 reflectDir = reflect(-lightDir, normal_);

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal_, halfwayDir), 0.0), 32.0);

    vec3 specular = vec3(0.3) * spec; // assuming bright white light color
    outColor = vec4(ambient + diffuse + specular, 1.0);
}

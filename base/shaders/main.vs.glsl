#version 140

in vec3 position;

uniform mat4 viewMat;
uniform mat4 projMat;
uniform mat4 modelMat;

void main()
{
    gl_Position = projMat * viewMat * modelMat * vec4(position, 1.0);
}
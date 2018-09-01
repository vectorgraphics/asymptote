//#version 140

in vec3 position;
in vec3 normal;

#ifdef EXPLICIT_COLOR
in vec4 color;
#endif

uniform mat4 viewMat;
uniform mat4 projMat;
uniform mat4 modelMat;

out vec3 Normal;

#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif


void main()
{
    gl_Position=projMat * viewMat * modelMat * vec4(position, 1.0);
    
    Normal=(transpose(inverse(viewMat * modelMat)) * vec4(normal,0)).xyz;

#ifdef EXPLICIT_COLOR
    Color=color;
#endif
}
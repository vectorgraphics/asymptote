//#version 450

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

mat4 invtransp(mat4 inmat)
{
    return transpose(inverse(inmat));
}

void main()
{
    gl_Position=projMat * viewMat * modelMat * vec4(position, 1.0);
    vec4 rawNormal=invtransp(viewMat * modelMat) * vec4(normal,0);

    Normal=normalize(rawNormal.xyz);
    
#ifdef EXPLICIT_COLOR
    Color=color;
#endif
}
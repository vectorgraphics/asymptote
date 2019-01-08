//#version 450

in vec3 position;
in vec3 normal;

#ifdef EXPLICIT_COLOR
in vec4 color;
#endif

uniform mat4 viewMat;
uniform mat4 projMat;

out vec3 Normal;
out vec3 ViewPosition;

#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif

mat4 invtransp(mat4 inmat)
{
    return transpose(inverse(inmat));
}

void main()
{
    gl_Position=projMat*viewMat*vec4(position,1.0);
    ViewPosition=(viewMat*vec4(position,1.0)).xyz;
    vec4 rawNormal=invtransp(viewMat)*vec4(normal,0);
    Normal=normalize(rawNormal.xyz);
    
#ifdef EXPLICIT_COLOR
    Color=color;
#endif
}
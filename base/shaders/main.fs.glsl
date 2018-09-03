//#version 450

struct Material
{
    vec4 diffuse, specular, emissive, ambient;
    float shininess; 
};

uniform Material materialData;

in vec3 Normal;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
#ifdef EXPLICIT_COLOR
    outColor=Color;
#else
    outColor=materialData.diffuse;
#endif

}

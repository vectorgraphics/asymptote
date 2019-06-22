// Alternative shader for outline mode.
struct Material
{
  vec4 diffuse,ambient,emissive,specular;
  vec4 parameters;
};

struct Light
{
  vec4 direction;
  vec4 diffuse,ambient,specular;  
};

uniform int nlights;
uniform Light lights[Nlights];

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

in vec3 vNormal;
#ifdef EXPLICIT_COLOR
in vec4 vColor; 
#endif

flat in int vMaterialIndex;

out vec4 outColor;

void main() {
    outColor = vec4(0,0,0,1);
}
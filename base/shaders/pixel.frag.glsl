struct Material
{
  vec4 diffuse,ambient,emissive,specular;
  vec4 parameters;
};

uniform MaterialBuffer
{
  Material Materials[Nmaterials];
};

out vec4 outColor;
flat in int materialIndex;

void main() {
    Material matobj = Materials[materialIndex];
    outColor = matobj.emissive;
}
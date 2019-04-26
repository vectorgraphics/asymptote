struct Material
{
  vec4 diffuse,ambient,emissive,specular;
  float shininess; 
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

in vec3 Normal;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif
flat in int materialIndex;

out vec4 outColor;

void main()
{
  vec4 Diffuse;
  vec4 Ambient;
  vec4 Emissive;
  vec4 Specular;
  float Shininess;

#ifdef EXPLICIT_COLOR
  if(materialIndex < 0) {
    int index=-materialIndex-1;
    Material m=Materials[index];
    Diffuse=Color;
    Ambient=Color;
    Emissive=vec4(0.0,0.0,0.0,1.0);
    Specular=m.specular;
    Shininess=m.shininess;
  } else {
    Material m=Materials[materialIndex];
    Diffuse=m.diffuse;
    Ambient=m.ambient;
    Emissive=m.emissive;
    Specular=m.specular;
    Shininess=m.shininess;
  }
#else
  Material m=Materials[materialIndex];
  Diffuse=m.diffuse;
  Ambient=m.ambient;
  Emissive=m.emissive;
  Specular=m.specular;
  Shininess=m.shininess;
#endif
  // Phong-Blinn model
  if(nlights > 0) {
    vec3 diffuse=vec3(0,0,0);
    vec3 specular=vec3(0,0,0);
    vec3 ambient=vec3(0,0,0);
    vec3 Z=vec3(0,0,1);
        
    for(int i=0; i < nlights; ++i) {
      vec3 L=normalize(lights[i].direction.xyz);
      diffuse += lights[i].diffuse.rgb*abs(dot(Normal,L));
      ambient += lights[i].ambient.rgb;
      specular += pow(abs(dot(Normal,normalize(L+Z))),Shininess)*
        lights[i].specular.rgb;
    }

    vec3 color=diffuse*Diffuse.rgb+
      ambient*Ambient.rgb+
      specular*Specular.rgb+
      Emissive.rgb;
    outColor=vec4(color,Diffuse[3]);
  } else
    outColor=Diffuse;
}

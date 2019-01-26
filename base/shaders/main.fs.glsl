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

uniform Light lights[Nlights];

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};


in vec3 Normal;
in float materialIndex;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
  vec4 Diffuse;
  vec4 Ambient;
  vec4 Emissive;
  vec4 Specular;
  float Shininess;

#ifdef EXPLICIT_COLOR
  Diffuse=Color;
  Ambient=Color;
  Emissive=vec4(0.0,0.0,0.0,1.0);
  Specular=vec4(0.75,0.75,0.75,1.0);
  Shininess=32;
#else
  int imaterial=int(materialIndex+0.5);
  Material m=Materials[imaterial];
  Diffuse=m.diffuse;
  Ambient=m.ambient;
  Emissive=m.emissive;
  Specular=m.specular;
  Shininess=m.shininess;
#endif
  // Phong-Blinn model
  if(Nlights > 0) {
    vec3 diffuse=vec3(0,0,0);
    vec3 specular=vec3(0,0,0);
    vec3 ambient=vec3(0,0,0);
    vec3 Z=vec3(0,0,1);
        
    for(int i=0; i < Nlights; ++i) {
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
  } else {
    outColor=Diffuse;
  }
}

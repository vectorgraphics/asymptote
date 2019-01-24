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

uniform Material materialData;

in vec3 Normal;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
  vec4 Diffuse;
  vec4 Ambient;
  vec4 Emissive;
#ifdef EXPLICIT_COLOR
  Diffuse=Color;
  Ambient=Color;
  Emissive=vec4(0.0,0.0,0.0,1.0);
#else
  Diffuse=materialData.diffuse;
  Ambient=materialData.ambient;
  Emissive=materialData.emissive;
#endif
  // Phong-Blinn model
  if(Nlights > 0) {
    vec3 diffuse=vec3(0,0,0);
    vec3 specular=vec3(0,0,0);
    vec3 ambient=vec3(0,0,0);
    vec3 Z=vec3(0,0,1);
        
    for(int i=0; i < Nlights; ++i) {
      vec3 L=normalize(lights[i].direction.xyz);
      float dotproduct=abs(dot(Normal,L));
      diffuse += lights[i].diffuse.rgb*dotproduct;
      ambient += lights[i].ambient.rgb;
      dotproduct=abs(dot(Normal,normalize(L+Z)));
      specular += pow(dotproduct,materialData.shininess)*
        lights[i].specular.rgb;
    }

    vec3 color=diffuse*Diffuse.rgb+
      ambient*Ambient.rgb+
      specular*materialData.specular.rgb+
      Emissive.rgb;
    outColor=vec4(color,Diffuse[3]);
  } else {
    outColor=Diffuse;
  }
}

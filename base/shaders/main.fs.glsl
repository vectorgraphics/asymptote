struct Material
{
    vec4 diffuse, specular, emissive, ambient;
    float shininess; 
};

struct Light
{
    vec4 direction;
    vec4 diffuse, ambient, specular;  
};

layout(std430,binding=1) buffer data
{
 Light lights[];
};

uniform int Nlights;

uniform Material materialData;

in vec3 Normal;
in vec3 ViewPosition;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

void main()
{
#ifdef EXPLICIT_COLOR
    outColor=Color;
#else
    // TODO: Change this to a PBR model
    // ==> Diffuse, metallic, roughness, fresnelIOR 

    // for now, the old Phong-Blinn model.
    if(Nlights > 0) {
        vec3 diffuse=vec3(0,0,0);
        vec3 specular=vec3(0,0,0);
        vec3 ambient=vec3(0,0,0);
        vec3 Z=vec3(0,0,1);
        
        for(int i=0; i < Nlights; ++i) {
            vec3 L=normalize(lights[i].direction.xyz);
            float lambertPower=max(dot(Normal,L),0);
            diffuse += lights[i].diffuse.rgb*lambertPower;
            ambient += lights[i].ambient.rgb;
            float dotproduct=dot(Normal,normalize(L+Z));
            if(dotproduct > 0)
               specular += pow(dotproduct,materialData.shininess)*
                   lights[i].specular.rgb;
        }

        vec3 color=diffuse*materialData.diffuse.rgb+
             ambient*materialData.ambient.rgb
             +materialData.specular.rgb*specular
             +materialData.emissive.rgb;
        outColor=vec4(color,materialData.diffuse[3]);
    } else {
        outColor=materialData.diffuse;
    }
    
#endif

}

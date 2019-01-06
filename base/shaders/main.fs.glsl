//#version 450

struct Material
{
    vec4 diffuse, specular, emissive, ambient;
    float shininess; 
};

struct Light
{
    vec3 direction;
    vec4 diffuse, specular, ambient; 
};

/*
struct Material
{
    vec4 diffuse, normal, ambient;
    float metallic, roughness;
}
*/

/*
// FIXME: Add SSBO rather than hard light limit
layout(std430,binding=1) buffer lightData
{
    int numLights;
    Light lights[];   
};
*/
uniform int lightCount; 
uniform Light lights[100];// FIXME

uniform Material materialData;

// in mat4 invtranspViewMat;
in vec3 Normal;
in vec3 ViewPosition;

#ifdef EXPLICIT_COLOR
in vec4 Color; 
#endif

out vec4 outColor;

mat4 invtransp(mat4 inmat)
{
    return transpose(inverse(inmat));
}

void main()
{
#ifdef EXPLICIT_COLOR
    outColor=Color;
#else
    // TODO: Change this to a PBR model
    // ==> Diffuse, metallic, roughness, fresnelIOR 

    // for now, the old Phong-Blinn model.
    if(lightCount>0) {
        vec3 diffuse=vec3(0,0,0);
        vec3 specular=vec3(0,0,0);
        vec3 ambient=vec3(0,0,0);
        vec3 Z=vec3(0,0,1);
        
        for(int i=0; i < lightCount; ++i) {
            vec3 L=normalize(lights[i].direction);
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

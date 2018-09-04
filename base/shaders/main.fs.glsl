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
// FIXME: Add SSBO rather than hard light limit
layout(std430,binding=1) buffer lightData
{
    int numLights;
    Light lights[];   
};
*/
uniform int lightCount; 
uniform Light lights[100];

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
    // FIXME: CHange this to a PBR model
    // ==> Diffuse, metallic, roughness, fresnelIOR 

    // for now, the old Phong model.
    if(lightCount>0) {
        vec3 diffuse=vec3(0,0,0);
        vec3 specular=vec3(0,0,0);
        vec4 ambient=vec4(0,0,0,1);

        // FIXME: Surely, PBR fixes this problem by using reflection maps
        // does this get fixed by other shading models like Blinn-Phong?
        vec3 incidence=normalize(ViewPosition);
        
        for(int i=0;i<lightCount;++i)
        {
            vec4 viewLightDir=normalize(vec4(-lights[i].direction,0));

            float lambertPower=dot(Normal,-viewLightDir.xyz);
            lambertPower=clamp(lambertPower,0,1);
            vec3 rawDiffuse=lights[i].diffuse.rgb*lambertPower;
            diffuse+=clamp(rawDiffuse,0,1);

            vec3 reflVector=reflect(viewLightDir.xyz,Normal);

            float specularPower=pow(dot(-incidence,reflVector),materialData.shininess);
            specularPower=clamp(specularPower,0,1);
            vec3 rawSpec=lights[i].specular.rgb*specularPower; 
            specular+=clamp(rawSpec,0,1);
        }

        diffuse=clamp(diffuse,0,1);

        outColor=vec4(diffuse,1)*materialData.diffuse;
        outColor+=lights[0].ambient*materialData.ambient;
        outColor+=materialData.specular*vec4(specular,1);
        outColor+=materialData.emissive;

        outColor=clamp(outColor,0,1);
    } else {
        outColor=materialData.diffuse;
    }
    
#endif

}

#version 450

struct Material
{
    vec4 diffuse, emissive, specular;
    vec4 parameters;
};

struct Light
{
    vec3 direction;
    vec3 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 normMat;
    vec3 viewPos;
} ubo;

layout(binding = 1, std430) buffer MaterialBuffer {
    Material materials[];
};

layout(binding = 2, std430) buffer LightBuffer {
    Light lights[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 norm;
layout(location = 2) flat in int materialIndex;

layout(location = 0) out vec4 outColor;

vec3 Diffuse;
vec3 Emissive;
vec3 Specular;
float Metallic;
float Fresnel0;
float Roughness2;
float Roughness;

vec3 normal;

float NDF_TRG(vec3 h)
{
    float ndoth=max(dot(normal,h),0.0);
    float alpha2=Roughness2*Roughness2;
    float denom=ndoth*ndoth*(alpha2-1.0)+1.0;
    return denom != 0.0 ? alpha2/(denom*denom) : 0.0;
}

float GGX_Geom(vec3 v)
{
    float ndotv=max(dot(v,normal),0.0);
    float ap=1.0+Roughness2;
    float k=0.125*ap*ap;
    return ndotv/((ndotv*(1.0-k))+k);
}

float Geom(vec3 v, vec3 l)
{
    return GGX_Geom(v) * GGX_Geom(l);
}

float Fresnel(vec3 h, vec3 v, float fresnel0)
{
    float a=1.0-max(dot(h,v),0.0);
    float b=a*a;
    return fresnel0+(1.0-fresnel0)*b*b*a;
}

vec3 BRDF(vec3 viewDirection, vec3 lightDirection)
{
    vec3 lambertian=Diffuse;
    // Cook-Torrance model
    vec3 h=normalize(lightDirection+viewDirection);

    float omegain=max(dot(viewDirection,normal),0.0);
    float omegaln=max(dot(lightDirection,normal),0.0);

    float D=NDF_TRG(h);
    float G=Geom(viewDirection,lightDirection);
    float F=Fresnel(h,viewDirection,Fresnel0);

    float denom=4.0*omegain*omegaln;
    float rawReflectance=denom > 0.0 ? (D*G)/denom : 0.0;

    vec3 dielectric=mix(lambertian,rawReflectance*Specular,F);
    vec3 metal=rawReflectance*Diffuse;

    return mix(dielectric,metal,Metallic);
}

void main() {

    Material mat = materials[materialIndex];

    Diffuse = mat.diffuse.rgb;
    Emissive = mat.emissive.rgb;
    Specular = mat.specular.rgb;
    Roughness = 1.f - mat.parameters[0];
    Metallic = mat.parameters[1];
    Fresnel0 = mat.parameters[2];
    Roughness2 = Roughness * Roughness;

    vec3 viewDirection = normalize(ubo.viewPos - position);
    normal = normalize(norm);

    if (!gl_FrontFacing)
        normal = -normal;

    outColor = vec4(Emissive.rgb, 1.0);

    for (int i = 0; i < 1; i++)
    {
        Light light = lights[i];

        float radiance = max(dot(normal, light.direction), 0.0);
        outColor += vec4(BRDF(viewDirection, light.direction) * radiance, 0.0);
    }
}

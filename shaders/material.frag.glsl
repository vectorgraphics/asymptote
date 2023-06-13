#version 450
#define MATERIAL
#define NORMAL

struct Material
{
    vec4 diffuse, emissive, specular;
    vec4 parameters;
};

struct Light
{
    vec4 direction;
    vec4 color;
};

layout(binding = 0) uniform UniformBufferObject {
    mat4 projViewMat;
    mat4 viewMat;
    mat4 normMat;
} ubo;

layout(binding = 1, std430) buffer MaterialBuffer {
    Material materials[];
};

layout(binding = 2, std430) buffer LightBuffer {
    Light lights[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 viewPos;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec4 inColor;
layout(location = 4) flat in int materialIndex;

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
    // constants[0] = nlights
} push;

layout(location = 0) out vec4 outColor;

vec3 Emissive;
vec3 Diffuse;
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

    uint nlights = push.constants[0];

    Material mat;

#ifdef GENERAL
    mat = materials[abs(materialIndex) - 1];

    if (materialIndex < 0) {
        mat.diffuse = inColor;
#ifdef NOLIGHTS
        mat.emissive += inColor;
#endif /*NOLIGHTS*/
    }

#else

    mat = materials[materialIndex];

#ifdef COLOR
    mat.diffuse = inColor;
#endif /*COLOR*/
#endif /*GENERAL*/

    outColor = mat.emissive;

#ifdef NORMAL

    Diffuse = mat.diffuse.rgb;
    Specular = mat.specular.rgb;
    Roughness = 1.f - mat.parameters[0];
    Metallic = mat.parameters[1];
    Fresnel0 = mat.parameters[2];
    Roughness2 = Roughness * Roughness;

    vec3 viewDirection = -normalize(viewPos);
    normal = normalize(norm);

    if (!gl_FrontFacing)
        normal = -normal;

    for (int i = 0; i < nlights; i++)
    {
        Light light = lights[i];

        vec3 radiance = max(dot(normal, light.direction.xyz), 0.0) * light.color.rgb;
        outColor += vec4(BRDF(viewDirection, light.direction.xyz) * radiance, 0.0);
    }

    outColor = vec4(outColor.rgb, mat.diffuse.a);
#endif /*NORMAL*/
}

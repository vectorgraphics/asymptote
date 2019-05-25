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
in vec3 ViewPosition;

// TODO: Integrate these constants into asy side
// PBR material parameters
vec3 PBRBaseColor; // Diffuse for nonmetals, reflectance for metals.
vec3 PBRSpecular; // Specular tint for nonmetals
float PBRMetallic; // Metallic/Nonmetals switch flag
float PBRF0; // Fresnel at zero for nonmetals
float PBRRoughness; // Roughness.

// Here is a good paper on BRDF models...
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf

uniform sampler2D environmentMap;

#ifdef ENABLE_TEXTURE
// I don't think we can put acos(-1)... 
const float PI = 3.14159265359;
// TODO: Enable different samples:
const int numSamples = 5;

// (x,y,z) -> (r, theta, phi);
// theta -> [0,\pi], "height" angle
// phi -> [0, 2\pi], rotation agnle
vec3 cart2spher(vec3 cart) {
  float x = cart.z;
  float y = cart.x;
  float z = cart.y;

  float r = length(cart);
  float phi = atan(y,x);
  float theta = acos(z/r);

  return vec3(r,phi,theta);
}

vec2 normalizedAngle(vec3 cartVec) {
  vec3 sphericalVec = cart2spher(cartVec);
  sphericalVec.y = sphericalVec.y / (2 * PI) - 0.25;
  sphericalVec.z = sphericalVec.z / PI;
  // sphericalVec.z = - sphericalVec.z;
  return sphericalVec.yz;
}
#endif

// h is the halfway vector between normal and light direction
float NDF(vec3 h, float roughness) {
  float ndoth = abs(dot(Normal, h));
  float alpha2 = roughness * roughness;

  float denom = pow((pow(ndoth,2) * (alpha2-1)) + 1, 2);
  return alpha2/denom;
}

float GGX(vec3 v) {
  float ndotv = abs(dot(v,Normal));
  float ap = pow((1+PBRRoughness),2);
  float k = ap/8;

  return ndotv/((ndotv * (1-k)) + k);
}

float Geom(vec3 v, vec3 l) {
  return GGX(v) * GGX(l);
}

// Schlick's approximation
float Fresnel(vec3 h, vec3 v, float F0) {
  float hdotv = max(dot(h,v), 0.0);
  
  return F0 + (1-F0)*pow((1-hdotv),5);
}


vec3 BRDF(vec3 viewDirection, vec3 lightDirection) {
  // Lambertian diffuse 
  vec3 lambertian = PBRBaseColor;
  // Cook-Torrance model
  vec3 h = normalize(lightDirection + viewDirection);


  float omegain = abs(dot(viewDirection, Normal));
  float omegaln = dot(lightDirection, Normal);

  // correctly flip the normal sign.
  vec3 hn = vec3(0);
  if (omegaln >= 0) {
    hn = normalize(Normal + lightDirection);
  } else {
    omegaln = -omegaln;
    hn = normalize(-Normal + lightDirection);
  }

  float D = NDF(hn, PBRRoughness);
  float G = Geom(viewDirection, lightDirection);
  float F = Fresnel(h, viewDirection, PBRF0);

  float rawReflectance = (D*G)/(4 * omegain * omegaln);

  vec3 dielectric = mix(lambertian, rawReflectance * PBRSpecular, F);
  vec3 metal = rawReflectance * PBRBaseColor;
  
  return mix(dielectric, metal, PBRMetallic);
}

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
  Shininess=m.shininess/128;

  PBRMetallic = 0;
  PBRBaseColor = Diffuse.rgb;
  PBRRoughness = 1 - Shininess;
  PBRF0 = 0.04; // Allow for Custom hardcoding in the future?
  PBRSpecular = Specular.rgb;
#endif

  if(nlights > 0) {
    vec3 pointLightRadiance=vec3(0,0,0);
    vec3 Z=vec3(0,0,1);
    
    // Formally, the formula given a point x and direction \omega,
    // L_i = \int_{\Omega} f(x, \omega_i, \omega) L(x,\omega_i) (\hat{n}\cdot \omega_i) \dif \omega_i
    // where \Omega is the hemisphere covering a point, f is the BRDF function
    // L is the radiance from a given angle and position.

    // as a finite point light, we have some simplification to the rendering equation.
    for(int i=0; i < nlights; ++i) {

      vec3 L = normalize(lights[i].direction.xyz);
      float cosTheta = abs(dot(Normal, normalize(lights[i].direction.xyz))); // $\omega_i \cdot n$ term
      float attn = 1; // if we have a good light direction.
      vec3 radiance = cosTheta * attn * lights[i].diffuse.rgb;

      // in our case, the viewing angle is always (0,0,1)... 
      // though the viewing angle does not matter in the Lambertian diffuse... 

      // totalRadiance += vec3(0,1,0)*Shininess;
      pointLightRadiance += BRDF(Z, L) * radiance;
    }

#ifdef ENABLE_TEXTURE
    // environment radiance -- riemann sums, for now.
    // can also do importance sampling
    vec3 envRadiance=vec3(0,0,0);

    vec3 normalPerp = vec3(-Normal.y, Normal.x, 0);
    if (length(normalPerp) == 0) { // x, y = 0.

      normalPerp = vec3(1, 0, 0);
    }
    // we now have a normal basis for Normal;
    // Normal;
    normalPerp = normalize(normalPerp);
    vec3 normalPerp2 = normalize(cross(Normal, normalPerp));

    bool useEnvironment = true;

    if (useEnvironment == true) {
      for (int ix=0; ix<numSamples; ++ix) {
        float x = ix/float(numSamples);
        for (int iy=0; iy<numSamples; ++iy) {
          float y = iy/float(numSamples);

          vec3 azimuth = sin(2*PI*x)*normalPerp + cos(2*PI*x)*normalPerp2;
          vec3 L = cos(PI*y/2)*azimuth + sin(PI*y/2)*Normal;

          vec3 rawRadiance = texture(environmentMap, normalizedAngle(L)).rgb;
          vec3 surfRefl = BRDF(Z, L);
          envRadiance += (surfRefl * rawRadiance * max(sin(PI*y),0))/(numSamples * numSamples);
        }
      }
    }

    envRadiance = (PI/2) * envRadiance;
  #endif
    vec3 color = Emissive.rgb + pointLightRadiance.rgb;

  #ifdef ENABLE_TEXTURE
    // vec3 lightVector = normalize(reflect(-Z, Normal));
    // vec2 anglemap = normalizedAngle(lightVector);
    // vec3 color = texture(environmentMap, anglemap).rgb;
    color += envRadiance.rgb;
  #endif 

    outColor=vec4(color,1);
  } else
    outColor=Diffuse;
}


in vec3 position;

#ifdef NORMAL

uniform mat4 viewMat;
out vec3 ViewPosition;

// orthographic projection; set at runtime, so switching projection types
// needs no shader recompilation
uniform bool orthographic;

uniform mat3 normMat;
in vec3 normal;
out vec3 Normal;

#endif

#ifdef MATERIAL
in int material;
flat out int materialIndex;
#endif

#ifdef COLOR
in vec4 color;
#endif

#ifdef WIDTH
in float width;
#endif

uniform mat4 projViewMat;

// runtime light count (same uniform as in the fragment shader): unlit
// scenes, including outline mode's runtime nlights=0, need no recompilation
uniform uint nlights;

#ifdef NORMAL
struct Material
{
  vec4 diffuse,emissive,specular;
  vec4 parameters;
};

uniform MaterialBuffer {
  Material Materials[Nmaterials];
};

out vec4 diffuse;
out vec3 specular;
out float Roughness2_in,Roughness_in,Metallic_in,Fresnel0_in;
out vec4 emissive;
#endif

void main()
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;
#ifdef NORMAL
  // In orthographic mode the eye is at infinity along view-z, so every
  // pixel's view direction is the constant (0,0,1); writing
  // ViewPosition=(0,0,-1) makes the fragment shader's
  // -normalize(ViewPosition) produce exactly that, keeping the fragment
  // shader branch-free. (The branch below is uniform; the matmul only
  // executes in perspective mode.)
  if(orthographic)
    ViewPosition=vec3(0.0,0.0,-1.0);
  else
    ViewPosition=(viewMat*v).xyz;
  Normal=normalize(normal*normMat);

  Material m;
#ifdef GENERAL
  materialIndex=material;
  m=Materials[abs(material)-1];
  emissive=m.emissive;
  if(material >= 0)
    diffuse=m.diffuse;
  else {
    if (m.parameters[3] != 0) {
      diffuse=color;
      // with no active lights the fragment shader's BRDF loop runs zero
      // times, so the color must reach the output via emissive
      if(nlights == 0u)
        emissive += color;
    } else {
      emissive += color;
      diffuse = m.diffuse;
    }
  }
#else
  materialIndex=material;
  m=Materials[material];
  emissive=m.emissive;
#ifdef COLOR
  if (m.parameters[3] != 0) {
    diffuse=color;
    // with no active lights the fragment shader's BRDF loop runs zero
    // times, so the color must reach the output via emissive
    if(nlights == 0u)
      emissive += color;
  } else {
    emissive += color;
    diffuse = m.diffuse;
  }
#else
  diffuse=m.diffuse;
#endif
#endif
  specular=m.specular.rgb;
  vec4 parameters=m.parameters;
  Roughness_in=1.0-parameters[0];
  Roughness2_in=Roughness_in*Roughness_in;
  Metallic_in=parameters[1];
  Fresnel0_in=parameters[2];
#else
#ifdef MATERIAL
  materialIndex=material;
#endif
#endif

#ifdef WIDTH
  gl_PointSize=width;
#endif
}

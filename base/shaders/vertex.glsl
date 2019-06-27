in vec3 position;
in vec3 normal;

#ifdef EXPLICIT_COLOR
in uint color;
#endif

in int material;

uniform mat4 projViewMat;
uniform mat4 viewMat;
uniform mat4 normMat;

// out vec3 ViewPosition;
out vec3 vNormal;
    
#ifdef EXPLICIT_COLOR
out vec4 vColor;
#endif

flat out int vMaterialIndex;

void main()
{
  gl_Position=projViewMat*vec4(position,1.0);
  // ViewPosition=(viewMat*vec4(position,1.0)).xyz;
  vNormal=(normMat*vec4(normal,0)).xyz;

#ifdef EXPLICIT_COLOR
  vColor=unpackUnorm4x8(color);
#endif

  vMaterialIndex=material;
}

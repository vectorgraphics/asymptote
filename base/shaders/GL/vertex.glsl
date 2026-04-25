in vec3 position;

#ifdef NORMAL

#ifndef ORTHOGRAPHIC
uniform mat4 viewMat;
out vec3 ViewPosition;
#endif

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
out vec4 Color;
#endif

#ifdef WIDTH
in float width;
#endif

uniform mat4 projViewMat;

void main()
{
  vec4 v=vec4(position,1.0);
  gl_Position=projViewMat*v;
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  ViewPosition=(viewMat*v).xyz;
#endif
  Normal=normalize(normal*normMat);
#endif

#ifdef COLOR
  Color=color;
#endif

#ifdef WIDTH
  gl_PointSize=width;
#endif

#ifdef MATERIAL
  materialIndex=material;
#endif
}

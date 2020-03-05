in vec3 position;

uniform mat3 normMat;

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
out vec3 ViewPosition;
#endif
in vec3 normal;
out vec3 Normal;
#endif

in int material;

#ifdef COLOR
in vec4 color;
out vec4 Color;
#endif

#ifdef WIDTH
in float width;
#endif

uniform mat4 projViewMat;
uniform mat4 viewMat;

flat out int materialIndex;

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

  materialIndex=material;
}

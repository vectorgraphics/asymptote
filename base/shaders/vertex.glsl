in vec3 position;

#ifdef NORMAL
in vec3 normal;
out vec3 Normal;
uniform mat3 normMat;
#else
#ifdef BILLBOARD
uniform mat3 normMat;
#endif
#endif

#ifdef EXPLICIT_COLOR
in vec4 color;
out vec4 Color;
#endif

#ifdef WIDTH
in float width;
#endif

in int material;

uniform mat4 projViewMat;
uniform mat4 viewMat;

out vec3 ViewPosition;

#ifdef BILLBOARD
in int center;
uniform vec3 Center[Ncenter];
#endif

flat out int materialIndex;

void main()
{
#ifdef BILLBOARD
  if(center > 0) {
    vec3 c=Center[center-1];
    gl_Position=projViewMat*vec4(c+(position-c)*normMat,1.0);
  } else
#endif
  gl_Position=projViewMat*vec4(position,1.0);
  ViewPosition=(viewMat*vec4(position,1.0)).xyz;
#ifdef NORMAL
  Normal=normMat*normal;
#endif

#ifdef EXPLICIT_COLOR
  Color=color;
#endif

#ifdef WIDTH
  gl_PointSize=width;
#endif

  materialIndex=material;
}

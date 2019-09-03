in vec3 position;

#ifdef NORMAL
in vec3 normal;
out vec3 Normal;
uniform mat4 normMat;
#endif

#ifdef EXPLICIT_COLOR
in uint color;
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
uniform mat3 billboardMat;
uniform vec3 Center[Ncenter];
#endif

flat out int materialIndex;

void main()
{
#ifdef BILLBOARD
  gl_Position=projViewMat*vec4(position,1.0);
  if(center > 0 && center <= 100) {
    vec3 c=Center[center-1];
    gl_Position=projViewMat*vec4(c+billboardMat*(position-c),1.0);
  } else
#else
  gl_Position=projViewMat*vec4(position,1.0);
#endif
  ViewPosition=(viewMat*vec4(position,1.0)).xyz;
#ifdef NORMAL
  Normal=(normMat*vec4(normal,0)).xyz;
#endif

#ifdef EXPLICIT_COLOR
  Color=unpackUnorm4x8(color);
#endif

#ifdef WIDTH
  gl_PointSize=width;
#endif

  materialIndex=material;
}

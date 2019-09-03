in vec3 position;

#ifdef NORMAL
in vec3 normal;
#endif

#ifdef EXPLICIT_COLOR
in uint color;
#endif

#ifdef BILLBOARD
in int center;
#endif

#ifdef WIDTH
in float width;
#endif

in int material;

uniform mat4 projViewMat;
uniform mat4 viewMat;
uniform mat4 normMat;
uniform mat3 billboardMat;

out vec3 ViewPosition;
#ifdef NORMAL
out vec3 Normal;
#endif
    
#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif

uniform vec3 Center[Ncenter];
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

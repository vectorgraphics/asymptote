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

out vec4 ViewPosition;

#ifdef BILLBOARD
in int centerIndex;
uniform vec3 Centers[Ncenter];
#endif

flat out int materialIndex;

void main()
{
#ifdef BILLBOARD
  int index=int(centerIndex);
  vec4 v=vec4(index == 0 ? position :
              Centers[index-1]+(position-Centers[index-1])*normMat,1.0);
#else    
  vec4 v=vec4(position,1.0);
#endif
  gl_Position=projViewMat*v;
  ViewPosition=viewMat*v;
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

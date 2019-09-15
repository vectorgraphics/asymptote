in vec3 position;

#ifdef NORMAL
#ifndef ORTHOGRAPHIC
out vec3 ViewPosition;
#endif
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
uniform mat3 viewMat;

#ifdef BILLBOARD
in int centerIndex;
uniform vec3 Centers[Ncenter];
#endif

flat out int materialIndex;

void main()
{
#ifdef BILLBOARD
  int index=int(centerIndex);
  vec3 v=index == 0 ? position :
    Centers[index-1]+(position-Centers[index-1])*normMat;
  gl_Position=projViewMat*vec4(v,1.0);
#else    
  vec3 v=position;
#endif
  gl_Position=projViewMat*vec4(v,1.0);
#ifdef NORMAL
#ifndef ORTHOGRAPHIC
  ViewPosition=viewMat*v;
#endif
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

in vec3 position;
in vec3 normal;

#ifdef EXPLICIT_COLOR
in vec4 color;
#endif

uniform mat4 projViewMat;
uniform mat4 viewMat;
uniform mat4 normMat;

out vec3 Normal;
out vec3 ViewPosition;

#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif

void main()
{
  gl_Position=projViewMat*vec4(position,1.0);
  ViewPosition=(viewMat*vec4(position,1.0)).xyz;
  Normal=normalize((normMat*vec4(normal,0)).xyz);

#ifdef EXPLICIT_COLOR
  Color=color;
#endif
}

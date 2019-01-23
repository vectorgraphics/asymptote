in vec3 position;
in vec3 normal;

#ifdef EXPLICIT_COLOR
in vec4 color;
#endif

uniform mat4 viewMat;
uniform mat4 projMat;

out vec3 Normal;
out vec3 ViewPosition;

#ifdef EXPLICIT_COLOR
out vec4 Color;
#endif

void main()
{
  gl_Position=projMat*viewMat*vec4(position,1.0);
  ViewPosition=(viewMat*vec4(position,1.0)).xyz;
  Normal=normalize((transpose(inverse(viewMat))*vec4(normal,0)).xyz);

#ifdef EXPLICIT_COLOR
  Color=color;
#endif
}

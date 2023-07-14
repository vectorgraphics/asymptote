#version 450
#define GPUINDEXING
#define HAVE_INTERLOCK
#define GPUCOMPRESS

void main()
{
  vec2 vertices[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1, 3));
  gl_Position = vec4(vertices[gl_VertexIndex],0,1);
}

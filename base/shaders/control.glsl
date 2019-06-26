layout (vertices = 10) out;

in vec3 WorldControls[];
out vec3 worldControls[];

in vec3 Normal[];
out vec3 normal[];

in int MaterialIndex[];
patch out int materialIndex;

void main()
{
  gl_TessLevelInner[0] = 1;
  gl_TessLevelOuter[0] = 1;
  gl_TessLevelOuter[1] = 1;
  gl_TessLevelOuter[2] = 1;

  gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;

  normal[gl_InvocationID]=Normal[gl_InvocationID];
  worldControls[gl_InvocationID]=WorldControls[gl_InvocationID];
  materialIndex=MaterialIndex[gl_InvocationID];
}

#version 450

layout(binding = 3, std430) buffer CountBuffer
{
  uint maxSize;
  uint index[];
};

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
    // constants[1] = width
} push;

layout(location = 0) out vec4 outColor;

void main()
{
  //index[uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x)]=1;
  atomicAdd(index[(push.constants[2] - uint(gl_FragCoord.y) - 1u)*push.constants[1]+uint(gl_FragCoord.x)],1u);
  discard;
}
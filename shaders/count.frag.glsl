#version 450

#ifdef GPUCOMPRESS
layout(binding=9, std430) buffer indexBuffer
{
  uint index[];
};
#else
layout(binding = 3, std430) buffer countBuffer
{
  uint maxSize;
  uint index[];
};
#endif

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
    // constants[1] = width
} push;

void main()
{
  atomicAdd(index[uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x)],1u);
  discard;
}
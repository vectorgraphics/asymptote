
layout(binding=3, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding=9, std430) buffer indexBuffer
{
  uint index[];
};

layout(binding=10, std430) buffer elementBuffer
{
  uint elements;
};

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
  // constants[1] = width
} push;

void main()
{
  uint pixel=uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x);
  uint Count=index[pixel];
  if(Count > 0u)
    count[(index[pixel]=atomicAdd(elements,1u))]=Count;

  discard;
}

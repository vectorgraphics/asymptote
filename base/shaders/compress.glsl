#ifdef GPUINDEXING
layout(binding=0, std430) buffer offsetBuffer
{
  uint offset[];
};
#define count offset
#else
layout(binding=2, std430) buffer countBuffer
{
  uint count[];
};
#endif

layout(binding=0) uniform atomic_uint elements;

layout(binding=1, std430) buffer indexBuffer
{
  uint index[];
};

uniform uint width;

void main()
{
  uint pixel=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  uint Count=index[pixel];
  if(Count > 0u)
    count[(index[pixel]=atomicCounterIncrement(elements))]=Count;
  discard;
}

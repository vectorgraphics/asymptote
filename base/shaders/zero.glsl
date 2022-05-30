layout(binding=2, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding=0) uniform atomic_uint elements;
uniform uint width;

void main()
{
  uint pixel=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  if(pixel == 0u) {
    maxSize=0u;
    atomicCounterExchange(elements,1u);
  }
  count[pixel]=0u;
  discard;
}

layout(binding=8, std430) buffer indexBuffer
{
  uint maxSize;
  uint index[];
};

uniform uint width;

void main()
{
  atomicAdd(index[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
  discard;
}

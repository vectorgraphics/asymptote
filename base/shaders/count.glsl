layout(binding=1, std430) buffer countBuffer {
  uint maxSize;
  uint count[];
};

uniform uint width;

void main()
{
  atomicAdd(count[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
  discard;
}

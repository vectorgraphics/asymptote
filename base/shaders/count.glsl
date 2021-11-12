layout(binding=2, std430) buffer countBuffer {
  uint count[];
};

uniform uint width;

void main()
{
  atomicAdd(count[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
  discard;
}

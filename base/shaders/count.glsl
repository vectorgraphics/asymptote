layout(binding=0) coherent buffer Count {
  uint count[];
};

uniform uint width;

void main()
{
  atomicAdd(count[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
}

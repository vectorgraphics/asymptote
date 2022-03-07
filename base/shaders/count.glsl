#ifdef GPUINDEXING
layout(binding=0, std430) buffer offsetBuffer {
  uint count[];
};
#else
layout(binding=2, std430) buffer countBuffer {
  uint count[];
};
#endif

uniform uint width;

void main()
{
  atomicAdd(count[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
  discard;
}

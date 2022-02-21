layout(binding=0, std430) buffer offsetBuffer {
  uint offset[];
};

uniform uint width;

void main()
{
  atomicAdd(offset[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)],1u);
  discard;
}

layout(binding=1, std430) buffer offsetBuffer {
  uint offset[];
};

uniform uint width;

void main()
{
  atomicAdd(offset[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)+1u],1u);
  discard;
}

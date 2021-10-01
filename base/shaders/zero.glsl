layout(binding=0) coherent buffer Counter {
  uint counter[];
};

uniform uint width;

void main()
{
  counter[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)]=0u;
  discard;
}

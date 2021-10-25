layout(binding=2) buffer Count {
  uint count[];
};

uniform uint width;

void main()
{
  count[uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x)]=0u;
  discard;
}

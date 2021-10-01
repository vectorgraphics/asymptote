layout(binding=0) coherent buffer Offset {
  uint offset[];
};

uniform uint width;

void main()
{
  uint headIndex=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  offset[headIndex]=0u;
  discard;
}

struct Fragment
{
  vec4 color;
  float depth;
};

layout(binding=0) coherent buffer Count {
  uint count[];
};

layout(binding=1) coherent buffer list {
  Fragment fragments[];
};

out vec4 outColor;

uniform uint width;

void main()
{
  uint headIndex=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  uint size=count[headIndex];
  uint listIndex=10u*headIndex;
  const uint maxSize=10u;
  size=min(maxSize,size);

  Fragment sortedList[maxSize];

  sortedList[0]=fragments[listIndex];
  // Sort the fragments in sortedList with respect to descending depth
  for(uint i=1u; i < size; i++) {
    Fragment temp=fragments[listIndex+i];
    uint j=i;
    while(j > 0u && temp.depth > sortedList[j-1u].depth) {
      sortedList[j]=sortedList[j-1u];
      j--;
    }
    sortedList[j]=temp;
  }

  outColor=sortedList[0].color;
  for(uint i=1u; i < size; i++)
    outColor=mix(outColor,sortedList[i].color,sortedList[i].color.a);
}

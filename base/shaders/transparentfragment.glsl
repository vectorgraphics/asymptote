struct Fragment
{
  vec4 color;
  float depth;
};

layout(binding=0) coherent buffer Count {
  uint count[];
};

layout(binding=1) coherent buffer Offset {
  uint offset[];
};

layout(binding=2) coherent buffer list {
  Fragment fragments[];
};

out vec4 outColor;

uniform uint width;
uniform vec4 background;

vec4 blend(vec4 outColor, vec4 color)
{
  return mix(outColor,color,color.a);
}

void main()
{
  uint headIndex=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  uint size=count[headIndex];
  if(size == 0u)
    discard;
  uint listIndex=offset[headIndex];
  const uint maxSize=10u;

  // Sort the fragments with respect to descending depth
  if(size < maxSize) {
    Fragment sortedList[maxSize];

    sortedList[0]=fragments[listIndex];
    for(uint i=1u; i < size; i++) {
      Fragment temp=fragments[listIndex+i];
      float depth=temp.depth;
      uint j=i;
      Fragment f;
      while(f=sortedList[j-1u], j > 0u && depth > f.depth) {
        sortedList[j]=f;
        j--;
      }
      sortedList[j]=temp;
    }

    outColor=background;
    for(uint i=0u; i < size; i++)
      outColor=blend(outColor,sortedList[i].color);
  } else {
    for(uint i=1u; i < size; i++) {
      Fragment temp=fragments[listIndex+i];
      float depth=temp.depth;
      uint j=i;
      Fragment f;
      while(f=fragments[listIndex+j-1u], j > 0u && depth > f.depth) {
        fragments[listIndex+j]=f;
        j--;
      }
      fragments[listIndex+j]=temp;
    }

    outColor=background;
    uint stop=listIndex+size;
    for(uint i=listIndex; i < stop; i++)
      outColor=blend(outColor,fragments[i].color);
  }
  count[headIndex]=0u;
}

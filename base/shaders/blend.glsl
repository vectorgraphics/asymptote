struct Fragment
{
  vec4 color;
  float depth;
};

layout(binding=1, std430) buffer offsetBuffer {
  uint offset[];
};

layout(binding=2, std430) buffer countBuffer {
  uint count[];
};

layout(binding=3, std430) buffer fragmentBuffer {
  Fragment fragment[];
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
  if(size == 0u) {
#ifdef GPUINDEXING
    offset[headIndex]=0u;
    discard;
#endif
  }
  uint listIndex=offset[headIndex];
  const uint maxSize=16u;

  // Sort the fragments with respect to descending depth
  if(size < maxSize) {
    Fragment sortedList[maxSize];

    sortedList[0]=fragment[listIndex];
    for(uint i=1u; i < size; i++) {
      Fragment temp=fragment[listIndex+i];
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
      Fragment temp=fragment[listIndex+i];
      float depth=temp.depth;
      uint j=i;
      Fragment f;
      while(f=fragment[listIndex+j-1u], j > 0u && depth > f.depth) {
        fragment[listIndex+j]=f;
        j--;
      }
      fragment[listIndex+j]=temp;
    }

    outColor=background;
    uint stop=listIndex+size;
    for(uint i=listIndex; i < stop; i++)
      outColor=blend(outColor,fragment[i].color);
  }
  count[headIndex]=0u;
#ifdef GPUINDEXING
    offset[headIndex]=0u;
#endif
}

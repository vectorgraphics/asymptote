layout(binding=1, std430) buffer offsetBuffer {
  uint offset[];
};

layout(binding=2, std430) buffer countBuffer {
  uint count[];
};

layout(binding=3, std430) buffer fragmentBuffer {
  vec4 fragment[];
};

layout(binding=4, std430) buffer depthBuffer {
  float depth[];
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
#endif
    discard;
  }
  uint listIndex=offset[headIndex];
  const uint maxSize=16u;

  // Sort the fragments with respect to descending depth
  if(size < maxSize) {
    vec4 sortedList[maxSize];
    float sortedDepth[maxSize];

    sortedList[0]=fragment[listIndex];
    sortedDepth[0]=depth[listIndex];
    for(uint i=1u; i < size; i++) {
      float D=depth[listIndex+i];
      uint j=i;
      float d;
      while(j > 0u && D > sortedDepth[j-1u]) {
        sortedList[j]=sortedList[j-1u];
        sortedDepth[j]=sortedDepth[j-1u];
        --j;
      }
      sortedList[j]=fragment[listIndex+i];
      sortedDepth[j]=D;
    }

    outColor=background;
    for(uint i=0u; i < size; i++)
      outColor=blend(outColor,sortedList[i]);
  } else {
    for(uint i=1u; i < size; i++) {
      vec4 temp=fragment[listIndex+i];
      float D=depth[listIndex+i];
      uint j=i;
      while(j > 0u && D > depth[listIndex+j-1u]) {
        fragment[listIndex+j]=fragment[listIndex+j-1u];
        depth[listIndex+j]=depth[listIndex+j-1u];
        --j;
      }
      fragment[listIndex+j]=temp;
      depth[listIndex+j]=D;
    }

    outColor=background;
    uint stop=listIndex+size;
    for(uint i=listIndex; i < stop; i++)
      outColor=blend(outColor,fragment[i]);
  }
  count[headIndex]=0u;
#ifdef GPUINDEXING
  offset[headIndex]=0u;
#endif
}

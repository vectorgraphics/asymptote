layout(binding=0, std430) buffer sumBuffer {
  uint sum[];
};

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

layout(binding=5, std430) buffer opaqueBuffer {
  vec4 opaqueColor[];
};

layout(binding=6, std430) buffer opaqueDepthBuffer {
  float opaqueDepth[];
};

layout(binding=7, std430) buffer sum2Buffer {
  uint sum2[];
};

out vec4 outColor;

uniform uint width;
uniform uint M;
uniform uint M2;
uniform uint r;
uniform vec4 background;

vec4 blend(vec4 outColor, vec4 color)
{
  return mix(outColor,color,color.a);
}

void main()
{
  uint headIndex=uint(gl_FragCoord.y)*width+uint(gl_FragCoord.x);
  uint size=count[headIndex];
  float OpaqueDepth=opaqueDepth[headIndex];
  if(size == 0u) {
#ifdef GPUINDEXING
    offset[headIndex]=0u;
#endif
    opaqueDepth[headIndex]=0.0;
    discard;
  }

  outColor=OpaqueDepth != 0.0 ? opaqueColor[headIndex] : background;

#ifdef GPUINDEXING
  uint p=headIndex < r*(M+1u) ? headIndex/(M+1u) : (headIndex-r)/M;
  uint listIndex=sum[p]+sum2[p/M2]+
#else
  uint listIndex=
#endif
    offset[headIndex]-size;

  // Sort the fragments with respect to descending depth
  if(size < ARRAYSIZE) {
    vec4 sortedColor[ARRAYSIZE];
    float sortedDepth[ARRAYSIZE];

    uint k=0u;

    if(OpaqueDepth != 0.0)
      while(k < size && depth[listIndex+k] >= OpaqueDepth)
        ++k;

    uint i=0u;
    if(k < size) {
      sortedColor[0]=fragment[listIndex+k];
      sortedDepth[0]=depth[listIndex+k];
      ++k;
      i=1u;
      while(true) {
        if(OpaqueDepth != 0.0)
          while(k < size && depth[listIndex+k] >= OpaqueDepth)
            ++k;
        if(k == size) break;
        float D=depth[listIndex+k];
        uint j=i;
        float d;
        while(j > 0u && D > sortedDepth[j-1u]) {
          sortedColor[j]=sortedColor[j-1u];
          sortedDepth[j]=sortedDepth[j-1u];
          --j;
        }
        sortedColor[j]=fragment[listIndex+k];
        sortedDepth[j]=D;
        ++i;
        ++k;
      }
    }
    for(uint j=0u; j < i; ++j)
      outColor=blend(outColor,sortedColor[j]);
  } else {
    uint k=0u;
    if(OpaqueDepth != 0.0)
      while(k < size && depth[listIndex+k] >= OpaqueDepth)
        ++k;
    for(uint i=k+1u; i < size; i++) {
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

    uint stop=listIndex+size;
    if(OpaqueDepth == 0.0)
      for(uint i=listIndex+k; i < stop; i++)
        outColor=blend(outColor,fragment[i]);
    else
      for(uint i=listIndex+k; i < stop; i++) {
        if(depth[i] < OpaqueDepth)
          outColor=blend(outColor,fragment[i]);
    }
  }

  count[headIndex]=0u;
  opaqueDepth[headIndex]=0.0;
#ifdef GPUINDEXING
  offset[headIndex]=0u;
#endif
}

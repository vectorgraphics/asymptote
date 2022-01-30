layout(binding=1) buffer head
{
  uint tail[];
};

layout(binding=2, std430) buffer list
{
  vec4 fragment[];
};

layout(binding=3, std430) buffer depthBuffer
{
  float depth[];
};

layout(binding=4, std430) buffer nextBuffer
{
  uint next[];
};

layout(binding=5, std430) buffer opaqueBuffer
{
  vec4 opaqueColor[];
};

layout(binding=6, std430) buffer opaqueDepthBuffer
{
  float opaqueDepth[];
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
  uint listIndex=tail[headIndex];
  const uint maxSize=16u; // Must be constant
  vec4 sortedColor[maxSize];
  float sortedDepth[maxSize];

  float OpaqueDepth=opaqueDepth[headIndex];

  uint i=0u;
  if(OpaqueDepth != 0.0)
    while(listIndex > 0u && depth[listIndex] >= OpaqueDepth)
      listIndex=next[listIndex];
  if(listIndex > 0u) {
    sortedColor[0]=fragment[listIndex];
    sortedDepth[0]=depth[listIndex];
    i=1u;
    for (; (listIndex=next[listIndex]) > 0u && i < maxSize; ++i) {
      if(OpaqueDepth != 0.0)
        while(listIndex > 0u && depth[listIndex] >= OpaqueDepth)
          listIndex=next[listIndex];
      if(listIndex == 0u) break;
      uint j=i;
      float D=depth[listIndex];
      while(j > 0u && D > sortedDepth[j-1u]) {
        sortedColor[j] = sortedColor[j-1u];
        sortedDepth[j] = sortedDepth[j-1u];
        j--;
      }
      sortedColor[j]=fragment[listIndex];
      sortedDepth[j]=D;
    }
  }

  // Combine fragments

  outColor=OpaqueDepth != 0.0 ? opaqueColor[headIndex] : vec4(1);
  if(i > 0u) {
    for (uint k = 0u; k < i; ++k)
      outColor=blend(outColor,sortedColor[k]);
  }

  tail[headIndex]=0u;
  opaqueDepth[headIndex]=0.0;
}

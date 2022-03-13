struct Fragment
{
  uint next;
  float depth;
};

layout(binding=1, std430) buffer head
{
  uint tail[];
};

layout(binding=2, std430) buffer list
{
  vec4 fragmentColor[];
};

layout(binding=3, std430) buffer fragmentBuffer
{
  Fragment fragment[];
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
    while(listIndex > 0u && fragment[listIndex].depth >= OpaqueDepth)
      listIndex=fragment[listIndex].next;
  if(listIndex > 0u) {
    sortedColor[0]=fragmentColor[listIndex];
    sortedDepth[0]=fragment[listIndex].depth;
    i=1u;
    for (; (listIndex=fragment[listIndex].next) > 0u && i < maxSize; ++i) {
      if(OpaqueDepth != 0.0)
        while(listIndex > 0u && fragment[listIndex].depth >= OpaqueDepth)
          listIndex=fragment[listIndex].next;
      if(listIndex == 0u) break;
      uint j=i;
      float D=fragment[listIndex].depth;
      while(j > 0u && D > sortedDepth[j-1u]) {
        sortedColor[j] = sortedColor[j-1u];
        sortedDepth[j] = sortedDepth[j-1u];
        j--;
      }
      sortedColor[j]=fragmentColor[listIndex];
      sortedDepth[j]=D;
    }
  }

  // Combine fragments

  if(OpaqueDepth != 0.0) {
    opaqueDepth[headIndex]=0.0;
    outColor=opaqueColor[headIndex];
  } else
    outColor=background;
  if(i > 0u) {
    for (uint k = 0u; k < i; ++k)
      outColor=blend(outColor,sortedColor[k]);
  }

  tail[headIndex]=0u;
}

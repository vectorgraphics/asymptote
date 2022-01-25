struct Fragment
{
  vec4 color;
  uint next;
  float depth;
};

layout(binding=1) coherent buffer head {
  uint tail[];
};

layout(binding=2) coherent buffer list {
  Fragment fragments[];
};

struct OpaqueFragment
{
  vec4 color;
  float depth;
};
layout(binding=3) coherent buffer opaque {
  OpaqueFragment zbuffer[];
};

out vec4 outColor;

uniform uint width;

void main()
{
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  uint listIndex = tail[headIndex];
  const uint maxSize = 64u; // Must be constant
  Fragment sortedList[maxSize];
  uint sortedCount = 0u;

  // Insert fragments into sortedList (not yet sorted)
  for (; listIndex != uint(0) && sortedCount < maxSize; sortedCount++) {
    sortedList[sortedCount] = fragments[listIndex];
    listIndex = fragments[listIndex].next;
  }
  if (sortedCount == 0u) discard;

  // Sort the fragments in sortedList
  for (uint i = 1u; i < sortedCount; i++) {
    Fragment temp = sortedList[i];
    uint j = i;
    while(j > 0u && temp.depth > sortedList[j-1u].depth) {
      sortedList[j] = sortedList[j-uint(1)];
      j--;
    }
    sortedList[j] = temp;
  }

  // Combine fragments
  if (zbuffer[headIndex].depth != uint(0)) outColor = zbuffer[headIndex].color;
  else outColor = vec4(1);
  for (uint i = uint(0); i < sortedCount; i++)
    outColor = mix(outColor, sortedList[i].color, sortedList[i].color.a);

  tail[headIndex]=0u;
}

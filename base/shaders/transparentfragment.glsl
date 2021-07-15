struct Fragment
{
    uint next;
    vec4 color;
    float depth;
};
layout(std430, binding=1) coherent buffer head {
    uint tail[];
};
layout(std430, binding=2) coherent buffer list {
    Fragment fragments[];
};

struct OpaqueFragment
{
    vec4 color;
    float depth;
};
layout(std430, binding=3) coherent buffer opaque {
    OpaqueFragment zbuffer[];
};

out vec4 outColor;

uniform uint width;

void main()
{
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  uint listIndex = tail[headIndex];
  const uint maxSize = 1024; // Must be constant
  Fragment sortedList[maxSize];
  uint sortedCount = 0;
  
  // Insert fragments into sortedList (not yet sorted)
  for (uint i = 0; listIndex != 0 && i < maxSize; i++) {
      sortedList[sortedCount] = fragments[listIndex];
      listIndex = fragments[listIndex].next;
      sortedCount++;
  }

  // Sort the fragments in sortedList
 for (uint i = 1; i < sortedCount && i < maxSize; i++) {
    Fragment temp = sortedList[i];
    uint j = i;
    while(j > 0 && temp.depth > sortedList[j-1].depth) {
      sortedList[j] = sortedList[j-1];
      j--;
    }
    sortedList[j] = temp;
  }

  // Combine fragments
  vec4 frag = zbuffer[headIndex].color;
  if (zbuffer[headIndex].depth == 0)
    frag = sortedList[0].color;
  for (uint i = 0; i < sortedCount && i < maxSize; i++) {
    if (i != sortedCount-1 &&
        distance(sortedList[i].depth, sortedList[i+1].depth) < 0.001 &&
        distance(sortedList[i].color, sortedList[i+1].color) < 0.01)
      continue;
    frag = mix(frag, sortedList[i].color, sortedList[i].color.a);
  }
  outColor = frag;
}
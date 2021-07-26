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

// Manhattan distance
float mDistance(vec4 first, vec4 second) {
  return abs(first.r - second.r) +
         abs(first.g - second.g) +
         abs(first.b - second.b) +
         abs(first.a - second.a);
}

void main()
{
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  uint listIndex = tail[headIndex];
  const uint maxSize = 1024; // Must be constant
  Fragment sortedList[maxSize];
  uint sortedCount = 0;

  // Insert fragments into sortedList (not yet sorted)
  for (; listIndex != 0 && sortedCount < maxSize; sortedCount++) {
      sortedList[sortedCount] = fragments[listIndex];
      listIndex = fragments[listIndex].next;
  }
  if (sortedCount == 0) discard;

  // Sort the fragments in sortedList
 for (uint i = 1; i < sortedCount; i++) {
    Fragment temp = sortedList[i];
    uint j = i;
    while(j > 0 && temp.depth > sortedList[j-1].depth) {
      sortedList[j] = sortedList[j-1];
      j--;
    }
    sortedList[j] = temp;
  }

  // Combine fragments
  uint last = sortedCount - 1;
  if (zbuffer[headIndex].depth != 0)
    outColor = zbuffer[headIndex].color;
  else
    outColor = vec4(1);
  for (uint i = 0; i < last; i++) {
    if (sortedList[i].depth-sortedList[i+1].depth < 0.001 &&
        mDistance(sortedList[i].color, sortedList[i+1].color) < 0.01)
      continue;
    outColor = mix(outColor, sortedList[i].color, sortedList[i].color.a);
  }
  outColor = mix(outColor, sortedList[last].color, sortedList[last].color.a);
  outColor.a = outColor.a*sortedCount;
}

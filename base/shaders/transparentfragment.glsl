struct Fragment
{
    uint next;
    vec4 color;
    float depth;
};
layout(r32ui, binding=1) uniform coherent uimage2D counts;

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

// Manhattan distance
float mDistance(vec4 first, vec4 second) {
  return abs(first.r - second.r) +
         abs(first.g - second.g) +
         abs(first.b - second.b) +
         abs(first.a - second.a);
}

void main()
{
  return;
  uint headIndex = uint(gl_FragCoord.y) * width + uint(gl_FragCoord.x);
  uint size = imageLoad(counts, ivec2(gl_FragCoord.xy)).r;
  // if (size == 0) discard;
  const uint maxSize = uint(1024); // Must be constant
  size = min(maxSize, size);
  uint listIndex = headIndex*uint(10);
  Fragment sortedList[maxSize];

  // Insert fragments into sortedList (not yet sorted)
  for (uint i = uint(0); i < size; i++) {
      sortedList[i] = fragments[listIndex+i];
  }

  // Sort the fragments in sortedList
 for (uint i = uint(1); i < size; i++) {
    Fragment temp = sortedList[i];
    uint j = i;
    while(j > uint(0) && temp.depth > sortedList[j-uint(1)].depth) {
      sortedList[j] = sortedList[j-uint(1)];
      j--;
    }
    sortedList[j] = temp;
  }

  // Combine fragments
  if (zbuffer[headIndex].depth != uint(0)) outColor = zbuffer[headIndex].color;
  else outColor = vec4(1);
  for (uint i = uint(0); i < size; i++)
    outColor = mix(outColor, sortedList[i].color, sortedList[i].color.a);
}

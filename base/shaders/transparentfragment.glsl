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
  outColor = sortedList[0].color;
  for (uint i = 0; i < last; i++) {
    if (abs(sortedList[i].depth-sortedList[i+1].depth) < 0.001 &&
        distance(sortedList[i].color, sortedList[i+1].color) < 0.01)
      continue;
    outColor = mix(outColor, sortedList[i].color, sortedList[i].color.a);
    // outColor = vec4( (i/4)&1, (i/2)&1, (i)&1, 1);
  }
  outColor = mix(outColor, sortedList[last].color, sortedList[last].color.a);
  // outColor = vec4((sortedCount/4)&1, (sortedCount/2)&1, (sortedCount)&1, 1);
}

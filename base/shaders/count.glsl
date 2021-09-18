layout(r32ui, binding=0) uniform coherent uimage2D counts;

uniform uint width;
uniform uint height;

// out vec4 outColor;

void main()
{
  uint count = imageAtomicAdd(counts, ivec2(gl_FragCoord.xy), 1u);
}

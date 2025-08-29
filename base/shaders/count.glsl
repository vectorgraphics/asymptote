
#ifdef GPUCOMPRESS
layout(binding=9, std430) buffer indexBuffer
{
  uint index[];
};
#else
layout(binding = 3, std430) buffer countBuffer
{
  uint maxSize;
  uint index[];
};
#endif

layout(push_constant) uniform PushConstants
{
  uvec4 constants;
  // constants[0] = nlights
  // constants[1] = width;
  vec4 background;
} push;

void main()
{
  uint pixel=uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x);
  if(pixel >= push.constants[1]*push.constants[2]) discard;
  atomicAdd(index[pixel],1u);
  discard;
}

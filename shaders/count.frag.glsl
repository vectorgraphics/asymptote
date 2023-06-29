#version 450

layout(binding = 3, std430) buffer CountBuffer
{
  uint maxSize;
  uint index[];
};

layout(push_constant) uniform PushConstants
{
	uvec4 constants;
    // constants[1] = width
} push;

layout(location = 0) out vec4 outColor;

void main()
{
  uint pixel=uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x);
  index[pixel]=pixel;
  //atomicAdd(index[uint(gl_FragCoord.y)*push.constants[1]+uint(gl_FragCoord.x)],1u);
  
  outColor = vec4(gl_FragCoord.y / push.constants[1], maxSize/255.f, 0.f, 1.f);
}
#version 450

#define LOCALSIZE 256
#define BLOCKSIZE 8

layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

layout(push_constant) uniform PushConstants
{
	uint blockSize;
} push;

layout(binding=0, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding=1, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

layout(binding=3, std430) buffer feedbackBuffer
{
  uint size;
  uint fragments;
};

shared uint groupSum[LOCALSIZE];

void main()
{
  uint localSum[groupSize];
  uint id=gl_LocalInvocationID.x;

  uint dataOffset=push.blockSize*id;
  uint sum=0u;
  for(uint i=0; i < push.blockSize; ++i) {
    localSum[i]=sum;
    sum += globalSum[dataOffset+i];
  }

  groupSum[id]=sum;
  barrier();

  for(uint shift=1u; shift < LOCALSIZE; shift *= 2u) {
    uint read;
    if(shift <= id)
      read=groupSum[id]+groupSum[id-shift];
    barrier();
    if(shift <= id)
      groupSum[id]=read;
    barrier();
  }

  // shift local sums and store
  uint shift=id > 0u ? groupSum[id-1u] : 0u;
  for(uint i=0u; i < push.blockSize; ++i)
    globalSum[dataOffset+i]=localSum[i]+shift;
}

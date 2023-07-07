#version 450

#define LOCALSIZE 256
#define BLOCKSIZE 8

layout(local_size_x=LOCALSIZE) in;

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

const uint groupSize=LOCALSIZE*BLOCKSIZE;

void main()
{
  uint id=gl_LocalInvocationID.x;
  uint dataOffset=gl_WorkGroupID.x*groupSize+id;
  uint stop=dataOffset+groupSize;
  uint sum=0u;
  for(uint i=dataOffset; i < stop; i += LOCALSIZE)
    sum += count[i];

  groupSum[id]=sum;
  barrier();

  for(uint s=LOCALSIZE/2; s > 0u; s >>= 1u) {
    if(id < s)
      groupSum[id] += groupSum[id+s];
    barrier();
  }

  if(id == 0u)
    globalSum[gl_WorkGroupID.x]=groupSum[0u];
}

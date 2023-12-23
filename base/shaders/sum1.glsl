layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

uniform uint elements;

layout(binding=2, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

shared uint groupSum[LOCALSIZE];

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

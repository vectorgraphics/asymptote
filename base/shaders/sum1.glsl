layout(local_size_x=localSize) in;

const uint groupSize=localSize*blockSize;

layout(binding=2, std430) buffer countBuffer
{
  uint maxSize;
  uint count[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

// avoid bank conflicts and coalesce global memory accesses
shared uint groupSum[localSize+1u];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint dataOffset=gl_WorkGroupID.x*groupSize+id;
  uint sum=count[dataOffset];
  for(uint i=localSize; i < groupSize; i += localSize)
    sum += count[dataOffset+i];

  if(id == 0u)
    groupSum[0u]=0u;
  groupSum[id+1u]=sum;
  barrier();

  for(uint s=localSize/2; s > 0u; s >>= 1u) {
    if(id < s)
      groupSum[id] += groupSum[id+s];
    barrier();
  }

  if(id+1u == localSize)
    globalSum[gl_WorkGroupID.x+1u]=sum+groupSum[0u];
}

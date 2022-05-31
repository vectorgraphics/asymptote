layout(local_size_x=localSize) in;

const uint groupSize=localSize*blockSize;

layout(binding=0, std430) buffer offsetBuffer
{
  uint maxDepth;
  uint offset[];
};

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
shared uint shuffle[groupSize+localSize];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint dataOffset=gl_WorkGroupID.x*groupSize+id;
  uint shuffleOffset=id/blockSize+id;
  const uint stride=localSize/blockSize+localSize;
  for(uint i=0; i < blockSize; i++)
    shuffle[shuffleOffset+i*stride]=count[dataOffset+i*localSize];

  barrier();

  uint Offset=id*blockSize+id;
  uint stop=Offset+blockSize;
  uint sum=shuffle[Offset];
  for(uint i=Offset+1u; i < stop; ++i)
    shuffle[i]=sum += shuffle[i];

  if(id == 0u)
    groupSum[0u]=0u;
  groupSum[id+1u]=sum;
  barrier();

  for(uint shift=1u; shift < localSize; shift *= 2u) {
    uint read=id < shift ? groupSum[id] : groupSum[id]+groupSum[id-shift];
    barrier();
    groupSum[id]=read;
    barrier();
  }

  uint groupOffset=globalSum[gl_WorkGroupID.x];
  for(uint i=0u; i < blockSize; i++)
    offset[dataOffset+i*localSize]=
      shuffle[shuffleOffset+i*stride]+groupSum[(i*localSize+id)/blockSize]+
      groupOffset;
}

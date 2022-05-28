const uint groupSize=localSize*blockSize;

layout(local_size_x=localSize) in;

layout(binding=0, std430) buffer offsetBuffer
{
  uint offset[];
};

layout(binding=2, std430) buffer countBuffer
{
  uint count[];
};

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

layout(binding=7, std430) buffer opaqueDepthBuffer
{
  uint maxSize;
  float opaqueDepth[];
};

// avoid bank conflicts and coalesce global memory accesses
shared uint groupSum[localSize+1u];
shared uint shuffle[groupSize+localSize];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint start=gl_WorkGroupID.x*groupSize;

  uint dataOffset=start+id;
  uint shuffleOffset=id/blockSize+id;
  uint stride=localSize/blockSize+localSize;
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

  for(uint i=0u; i < blockSize; i++)
    offset[dataOffset+i*localSize]=
      shuffle[shuffleOffset+i*stride]+groupSum[(i*localSize+id)/blockSize];

  if(id+1u == localSize) {
    if(gl_WorkGroupID.x == 0u)
      globalSum[0u]=maxSize;
    globalSum[gl_WorkGroupID.x+1u]=
      shuffle[shuffleOffset+blockSize*stride-stride]+groupSum[localSize-1u];
  }
}

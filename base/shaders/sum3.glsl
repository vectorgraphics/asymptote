layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

uniform uint final;

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

layout(binding=8, std430) buffer feedbackBuffer
{
  uint size;
  uint fragments;
};

shared uint shuffle[groupSize+LOCALSIZE-1u];
shared uint groupSum[LOCALSIZE+1u];

void main()
{
  uint id=gl_LocalInvocationID.x;

// avoid bank conflicts and coalesce global memory accesses
  uint dataOffset=gl_WorkGroupID.x*groupSize+id;
  uint shuffleOffset=id/BLOCKSIZE+id;
  const uint stride=LOCALSIZE/BLOCKSIZE+LOCALSIZE;
  for(uint i=0u; i < BLOCKSIZE; i++)
    shuffle[shuffleOffset+i*stride]=count[dataOffset+i*LOCALSIZE];

  barrier();

  uint Offset=id*BLOCKSIZE+id;
  uint stop=Offset+BLOCKSIZE;

  uint sum=0u;
  for(uint i=Offset; i < stop; ++i)
    shuffle[i]=sum += shuffle[i];

  if(id == 0u)
    groupSum[0u]=0u;
  groupSum[id+1u]=sum;
  barrier();

  // Apply Hillis-Steele algorithm over all sums in workgroup
  for(uint shift=1u; shift < LOCALSIZE; shift *= 2u) {
    uint read;
    if(shift <= id)
      read=groupSum[id]+groupSum[id-shift];
    barrier();
    if(shift <= id)
      groupSum[id]=read;
    barrier();
  }

  uint groupOffset=globalSum[gl_WorkGroupID.x];
  for(uint i=0u; i < BLOCKSIZE; ++i)
    offset[dataOffset+i*LOCALSIZE]=shuffle[shuffleOffset+i*stride]+
      groupSum[(i*LOCALSIZE+id)/BLOCKSIZE]+groupOffset;

  uint diff=final-dataOffset;
  if(diff < groupSize && diff % LOCALSIZE == 0) {
    size=maxDepth;
    maxDepth=0u;
    fragments=offset[final+1u]=offset[final];
  }
}

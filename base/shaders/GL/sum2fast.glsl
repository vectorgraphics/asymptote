layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

layout(binding=8, std430) buffer feedbackBuffer
{
  uint maxSize;
  uint fragments;
};

shared uint shuffle[groupSize+LOCALSIZE-1u];
shared uint groupSum[LOCALSIZE+1u];

void main()
{
  uint id=gl_LocalInvocationID.x;

// avoid bank conflicts and coalesce global memory accesses
  uint shuffleOffset=id/BLOCKSIZE+id;
  const uint stride=LOCALSIZE/BLOCKSIZE+LOCALSIZE;
  for(uint i=0u; i < BLOCKSIZE; i++)
    shuffle[shuffleOffset+i*stride]=globalSum[id+i*LOCALSIZE];

  barrier();

  uint Offset=id*BLOCKSIZE+id;
  uint stop=Offset+BLOCKSIZE;
  uint sum=0u;
  for(uint i=Offset; i < stop; ++i) {
    uint value=shuffle[i];
    shuffle[i]=sum;
    sum += value;
  }

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

  for(uint i=0u; i < BLOCKSIZE; i++)
    globalSum[id+i*LOCALSIZE]=
      shuffle[shuffleOffset+i*stride]+groupSum[(i*LOCALSIZE+id)/BLOCKSIZE];
}

layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

layout(push_constant) uniform PushConstants
{
  uint blockSize;
  uint final;
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

layout(binding=2, std430) buffer offsetBuffer
{
  uint maxDepth;
  uint offset[];
};

layout(binding=3, std430) buffer feedbackBuffer
{
  uint size;
  uint fragments;
};

shared uint shuffle[groupSize+LOCALSIZE-1u];
shared uint groupSum[LOCALSIZE+1u];
shared uint maxSum[LOCALSIZE+1u]; // prefix max piggybacked on the sum reduction

void main()
{
  uint id=gl_LocalInvocationID.x;

// avoid bank conflicts and coalesce global memory accesses
  uint dataOffset=gl_WorkGroupID.x*groupSize+id;
  uint shuffleOffset=id/BLOCKSIZE+id;
  const uint stride=LOCALSIZE/BLOCKSIZE+LOCALSIZE;
  uint localMax=0u;
  for(uint i=0u; i < BLOCKSIZE; i++) {
    uint c=count[dataOffset+i*LOCALSIZE];
    shuffle[shuffleOffset+i*stride]=c;
    if(c > localMax) localMax=c;
  }

  barrier();

  uint Offset=id*BLOCKSIZE+id;
  uint stop=Offset+BLOCKSIZE;

  uint sum=0u;
  for(uint i=Offset; i < stop; ++i)
    shuffle[i]=sum += shuffle[i];

  // groupSum carries the per-thread block sums; maxSum carries the matching
  // per-thread block maxima through the same reduction (prefix max), so the
  // workgroup max of the per-pixel fragment counts adds no extra barriers.
  // It feeds the CPU's small/big blend-pipeline switch (resizeFragmentBuffer).
  if(id == 0u)
    {
      groupSum[0u]=0u;
      maxSum[0u]=0u;
    }
  groupSum[id+1u]=sum;
  maxSum[id+1u]=localMax;
  barrier();

  // Apply Hillis-Steele algorithm over all sums in workgroup
  for(uint shift=1u; shift < LOCALSIZE; shift *= 2u) {
    uint read, readMax;
    if(shift <= id)
      {
        read=groupSum[id]+groupSum[id-shift];
        readMax=max(maxSum[id],maxSum[id-shift]);
      }
    barrier();
    if(shift <= id)
      {
        groupSum[id]=read;
        maxSum[id]=readMax;
      }
    barrier();
  }
  // maxSum[LOCALSIZE-1] holds the workgroup max (prefix max over all threads)
  if(id == 0u)
    atomicMax(maxDepth,maxSum[LOCALSIZE-1u]);

  uint groupOffset=globalSum[gl_WorkGroupID.x];
  for(uint i=0u; i < BLOCKSIZE; ++i)
    offset[dataOffset+i*LOCALSIZE]=shuffle[shuffleOffset+i*stride]+
      groupSum[(i*LOCALSIZE+id)/BLOCKSIZE]+groupOffset;

  uint diff=push.final-dataOffset;
  if(diff < groupSize && diff % LOCALSIZE == 0) {
    // Atomic read+reset: other workgroups' atomicMax() calls may still be in
    // flight; any that land after this exchange carry over into the next
    // frame's snapshot (the switch logic tolerates one stale frame).
    size=atomicExchange(maxDepth,0u);
    fragments=offset[push.final+1u]=offset[push.final];
  }
}

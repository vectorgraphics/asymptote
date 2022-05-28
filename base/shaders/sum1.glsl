#define LOCALSIZE 256u
#define CHUNKSIZE 16u
const uint GROUPSIZE=LOCALSIZE*CHUNKSIZE;

layout(local_size_x=LOCALSIZE) in;

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
shared uint groupSum[LOCALSIZE+1u];
shared uint shuffle[GROUPSIZE+LOCALSIZE];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  uint start=gl_WorkGroupID.x*GROUPSIZE;

  uint dataOffset=start+id;
  uint shuffleOffset=id/CHUNKSIZE+id;
  uint stride=LOCALSIZE/CHUNKSIZE+LOCALSIZE;
  for(uint i=0; i < CHUNKSIZE; i++)
    shuffle[shuffleOffset+i*stride]=count[dataOffset+i*LOCALSIZE];

  barrier();

  uint Offset=id*CHUNKSIZE+id;
  uint stop=Offset+CHUNKSIZE;
  uint sum=shuffle[Offset];
  for(uint i=Offset+1u; i < stop; ++i)
    shuffle[i]=sum += shuffle[i];

  if(id == 0u)
    groupSum[0u]=0u;
  groupSum[id+1u]=sum;
  barrier();

  for(uint shift=1u; shift < LOCALSIZE; shift *= 2u) {
    uint read=id < shift ? groupSum[id] : groupSum[id]+groupSum[id-shift];
    barrier();
    groupSum[id]=read;
    barrier();
  }

  for(uint i=0u; i < CHUNKSIZE; i++)
    offset[dataOffset+i*LOCALSIZE]=
      shuffle[shuffleOffset+i*stride]+groupSum[(i*LOCALSIZE+id)/CHUNKSIZE];

  if(id+1u == LOCALSIZE) {
    if(gl_WorkGroupID.x == 0u)
      globalSum[0u]=maxSize;
    globalSum[gl_WorkGroupID.x+1u]=
      shuffle[shuffleOffset+CHUNKSIZE*stride-stride]+groupSum[LOCALSIZE-1u];
  }
}

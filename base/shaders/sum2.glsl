layout(local_size_x=LOCALSIZE) in;

const uint groupSize=LOCALSIZE*BLOCKSIZE;

uniform uint blockSize;

layout(binding=3, std430) buffer globalSumBuffer
{
  uint globalSum[];
};

shared uint groupSum[LOCALSIZE];

void main()
{
  uint localSum[groupSize];
  uint id=gl_LocalInvocationID.x;

  uint dataOffset=blockSize*id;
  uint sum=0u;
  for(uint i=0; i < blockSize; ++i) {
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
  for(uint i=0u; i < blockSize; ++i)
    globalSum[dataOffset+i]=localSum[i]+shift;
}

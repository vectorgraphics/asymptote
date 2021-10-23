layout(local_size_x=1024) in;

uniform uint nElements;

layout(binding=0) buffer Sum
{
  uint sum[];
};

layout(binding=1) buffer Data
{
  uint data[];
};

shared uint sharedData[gl_WorkGroupSize.x];

// Return ceil(log2(n))
uint ceillog2(uint n)
{
  const int MultiplyDeBruijnBitPosition[32]={
  0,9,1,10,13,21,2,29,11,14,16,18,22,25,3,30,
  8,12,20,28,15,17,24,7,19,27,23,6,26,5,4,31
  };
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;

  return MultiplyDeBruijnBitPosition[n*0x07C4ACDDU >> 27];
}

uint ceilquotient(uint a, uint b)
{
  return (a+b-1)/b;
}

void main(void)
{
  const uint id=gl_LocalInvocationID.x;
  sharedData[id]=sum[id];

  barrier();

  const uint steps=ceillog2(gl_WorkGroupSize.x); // Precompute this?

  for(uint step=0; step < steps; step++) {
    uint mask=1 << step;
    uint index=((id >> step) << (step+1))+mask;
    sharedData[index+(id&(mask-1))] += sharedData[index-1];
    barrier();
  }

  const uint m=ceilquotient(nElements,gl_WorkGroupSize.x);

  if(id+1 < gl_WorkGroupSize.x)
    data[m*(id+1)] += sharedData[id];
  else
    sum[0]=sharedData[id];  // Store fragment size in sum[0]
}

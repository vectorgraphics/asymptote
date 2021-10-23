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

uint ceilquotient(uint a, uint b)
{
  return (a+b-1)/b;
}

void main(void)
{
  const uint id=gl_LocalInvocationID.x;
  sharedData[id]=sum[id];

  barrier();

  const uint steps=10; // ceil(log2(gl_WorkGroupSize.x));

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

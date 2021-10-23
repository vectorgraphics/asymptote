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
  return (a+b-1u)/b;
}

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  sharedData[id]=sum[id];

  barrier();

  const uint steps=10u; // ceil(log2(gl_WorkGroupSize.x));

  for(uint step=0u; step < steps; step++) {
    uint mask=1u << step;
    uint index=((id >> step) << (step+1u))+mask;
    sharedData[index+(id&(mask-1u))] += sharedData[index-1u];
    barrier();
  }

  uint m=ceilquotient(nElements,gl_WorkGroupSize.x);

  if(id+1u < gl_WorkGroupSize.x)
    data[m*(id+1u)] += sharedData[id];
  else
    sum[0]=sharedData[id];  // Store fragment size in sum[0]
}

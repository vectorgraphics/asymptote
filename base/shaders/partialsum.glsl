layout(local_size_x=PROCESSORS) in;

uniform uint elements;
uniform uint steps;

layout(binding=0) buffer Sum
{
  uint sum[];
};

layout(binding=1) buffer Data
{
  uint data[];
};

shared uint sharedData[PROCESSORS];

uint ceilquotient(uint a, uint b)
{
  return (a+b-1u)/b;
}

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  sharedData[id]=sum[id];

  barrier();

  uint index=id << 1u;
  sharedData[index+1u] += sharedData[index];
  barrier();
  uint step;
  for(step=1u; step < steps-1u; step++) {
    uint mask=(1u << step)-1u;
    uint index=((id >> step) << (step+1u))+mask;
    uint windex=index+(id&mask)+1u;
    sharedData[windex] += sharedData[index];
    barrier();
  }
  uint mask=(1u << step)-1u;
  index=((id >> step) << steps)+mask;
  uint windex=index+(id&mask)+1u;
  if(windex < PROCESSORS)
    sharedData[windex] += sharedData[index];
  barrier();

  uint m=ceilquotient(elements,PROCESSORS);

  if(id+1u < PROCESSORS)
    data[m*(id+1u)] += sharedData[id];
  else
    sum[0]=sharedData[id];  // Store fragment size in sum[0]
}

layout(local_size_x=PROCESSORS) in;

uniform uint elements;

layout(binding=0, std430) buffer sumBuffer
{
  uint sum[];
};

shared uint sharedData[PROCESSORS];

void main(void)
{
  uint id=gl_LocalInvocationID.x;
  sharedData[id]=sum[id+1u];

  barrier();

  uint index=id << 1u;
  sharedData[index+1u] += sharedData[index];
  barrier();
  for(uint step=1u; step < STEPSM1; step++) {
    uint mask=(1u << step)-1u;
    uint index=((id >> step) << (step+1u))+mask;
    uint windex=index+(id&mask)+1u;
    sharedData[windex] += sharedData[index];
    barrier();
  }
  uint mask=(1u << STEPSM1)-1u;
  index=((id >> STEPSM1) << (STEPSM1+1u))+mask;
  uint windex=index+(id&mask)+1u;
  if(windex < PROCESSORS)
    sharedData[windex] += sharedData[index];
  barrier();

  sum[id+1u]=sharedData[id];
}

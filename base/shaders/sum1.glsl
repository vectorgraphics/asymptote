layout(local_size_x=LOCAL_SIZE_X) in;

uniform uint elements;

layout(binding=0, std430) buffer offsetBuffer
{
  uint offset[];
};

layout(binding=1, std430) buffer sumBuffer
{
  uint sum[];
};

void main(void)
{
  uint id=gl_GlobalInvocationID.x;

  uint m=elements/(gl_WorkGroupSize.x*gl_NumWorkGroups.x);
  uint r=elements-m*gl_WorkGroupSize.x*gl_NumWorkGroups.x;
  uint row,stop;
  if(id < r) {
    row=m*id+id;
    stop=row+m+1u;
  } else {
    row=m*id+r;
    stop=row+m;
  }

  uint Sum=offset[elements+row];
  offset[row]=Sum;
  for(uint i=row+1u; i < stop; ++i) {
    Sum += offset[elements+i];
    offset[i]=Sum;
  }

  sum[id+1u]=Sum;
}

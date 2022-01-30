layout(binding=0) uniform atomic_uint counter;

void main()
{
  atomicCounterIncrement(counter);
}

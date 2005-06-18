for (int i = 0; i < 1e5; ++i)
{
  file a=output((string)i);
  write(a,i,endl);
  close(a);
}
write("Pass.");


for (int i = 0; i < 1e7; ++i)
{
  file a=output("out/"+(string) i);
  write(a,i,endl);
  close(a);
}


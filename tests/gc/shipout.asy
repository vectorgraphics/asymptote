guide a;
for (int i = 0; i < 1e7; ++i)
{
  picture pic;
  size(pic,10cm,20cm);
  path p=(1.0,2.0);
  path q=(3.0,4.0);
  path a=p..q..(3,5);
  draw(pic,a,blue+dashed);
  shipout("out/",pic);
}

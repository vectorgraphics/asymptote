guide a;
for (int i = 0; i < 1e7; ++i)
{
  picture pic;
  size(pic,10cm,20cm);
  draw(pic,"a",scale(2)*(0,0)--(1,1));
  shipout("out/",pic);
}

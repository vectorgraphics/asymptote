for (int i = 0; i < 1e7; ++i)
{
  picture pic;
  unitsize(pic, 0.02mm, 0.04mm);
  draw(pic,unitcircle);
  shipout("out/",pic);
}

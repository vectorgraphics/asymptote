// Avoid two parentheses.
transform shift(real x, real y)
{
  return shift((x,y));
}

// A rotation in the direction dir limited to [-90,90]
// This is useful for rotating text along a line in the direction dir.
transform rotate(explicit pair dir)
{
  real angle=degrees(dir);
  if(angle > 90 && angle < 270) angle -= 180;
  return rotate(angle);
} 

transform shift(transform t)
{
  return (t.x,t.y,0,0,0,0);
}

transform shiftless(transform t)
{
  return (0,0,t.xx,t.xy,t.yx,t.yy);
}

transform scale(transform t)
{
  transform t0=shiftless(t);
  return rotate(-degrees(t0*(1,0)))*t0;
}

transform invert=reflect((0,0),(1,0));

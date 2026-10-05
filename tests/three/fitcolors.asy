import TestLib;
import three;
import palette;

// The largest difference between two pens in any rgb component or in opacity.
real distance(pen a, pen b)
{
  return max(max(abs(colors(rgb(a))-colors(rgb(b)))),
             abs(opacity(a)-opacity(b)));
}

pen at(vertexPen vp, triple z)
{
  return vp(surfaceVertex(z));
}

// The fit is regularized, so it reproduces the given colors only
// approximately.
real fitTolerance=1e-3;

// The largest error of the fit to colors at T*coords, at the data points.
real fitError(triple[] coords, pen[] colors, transform3 T=identity4)
{
  vertexPen vp=fitColors(T*coords,colors);
  real e=0;
  for(int i=0; i < coords.length; ++i)
    e=max(e,distance(at(vp,T*coords[i]),colors[i]));
  return e;
}

triple[] box;
for(int i=0; i < 2; ++i)
  for(int j=0; j < 2; ++j)
    for(int k=0; k < 2; ++k)
      box.push((i,j,k));
pen[] boxColors={red,green,blue,cyan,magenta,yellow,black,white};

triple[] square={(0,0,0),(1,0,0),(1,1,0),(0,1,0)};
pen[] squareColors={red,green,blue,yellow};

// Neither aligned with the axes nor centered at the origin.
transform3 T=shift(3,-2,5)*rotate(37,(1,2,3))*scale3(4);

triple[] samples={(0.5,0.5,0.5),(0.2,0.7,0.9),(0,0.5,1),(0.3,0.3,0.3)};

StartTest("fitColors: corners of a box");
{
  assert(fitError(box,boxColors) < fitTolerance);
  assert(fitError(box,boxColors,T) < fitTolerance);

  // The fit commutes with a rotation, shift, and scaling of the data.
  vertexPen vp=fitColors(box,boxColors);
  vertexPen vpT=fitColors(T*box,boxColors);
  for(triple z : samples)
    assert(distance(at(vp,z),at(vpT,T*z)) < 1e-8);

  // Colors that are linear in the position are fit by a linear function,
  // not by one of the cubics that agree with it at the corners.
  pen linear(triple z) {return rgb(z.x,z.y,z.z);}
  vertexPen vp=fitColors(T*box,sequence(new pen(int i) {
        return linear(box[i]);
      },box.length));
  for(triple z : samples)
    assert(distance(at(vp,T*z),linear(z)) < 0.01);
}
EndTest();

StartTest("fitColors: corners of a square");
{
  assert(fitError(square,squareColors) < fitTolerance);
  assert(fitError(square,squareColors,T) < fitTolerance);

  vertexPen vp=fitColors(square,squareColors);
  vertexPen vpT=fitColors(T*square,squareColors);
  for(triple z : samples)
    assert(distance(at(vp,z),at(vpT,T*z)) < 1e-8);

  // The colors are blended between the corners.
  assert(distance(at(vp,(0.5,0,0)),rgb(0.5,0.5,0)) < 0.01);
  assert(distance(at(vp,(0.5,0.5,0)),rgb(0.5,0.5,0.25)) < 0.01);
}
EndTest();

StartTest("fitColors: opacity");
{
  // Opaque data gives an exactly opaque result.
  vertexPen vp=fitColors(box,boxColors);
  for(triple z : samples)
    assert(opacity(at(vp,z)) == 1);

  // Otherwise the opacity is fit like a color channel.
  pen[] p=copy(squareColors);
  p[2]=blue+opacity(0.2);
  assert(fitError(square,p) < fitTolerance);
  assert(fitError(square,p,T) < fitTolerance);
  vertexPen vp=fitColors(square,p);
  assert(abs(opacity(at(vp,(0.5,0.5,0)))-0.8) < 0.01);

  // The blend mode is that of the first pen.
  p[0]=red+opacity(1,"Screen");
  p[2]=blue+opacity(0.2,"Multiply");
  assert(blend(at(fitColors(square,p),(0.5,0.5,0))) == "Screen");
}
EndTest();

StartTest("fitColors: colorspace and range");
{
  // The pens are converted to the colorspace of the first one.
  vertexPen vp=fitColors(square,new pen[] {cmyk(red),green,blue,gray(0.5)});
  assert(colorspace(at(vp,(0.5,0.5,0))) == "cmyk");
  assert(distance(at(vp,square[1]),green) < fitTolerance);

  vertexPen vp=fitColors(square,new pen[] {gray(0.2),gray(0.4),white,black});
  assert(colorspace(at(vp,(0.5,0.5,0))) == "gray");
  assert(distance(at(vp,square[2]),white) < fitTolerance);

  // Far from the data the components are clamped to [0,1].
  vertexPen vp=fitColors(box,boxColors);
  for(triple z : new triple[] {(100,-50,30),(-100,-100,-100),(0,0,1e6)}) {
    real[] c=colors(at(vp,z));
    assert(c.length == 3);
    assert(min(c) >= 0 && max(c) <= 1);
  }

  // The arrays passed in are left alone.
  triple[] coords=copy(square);
  pen[] p=copy(squareColors);
  fitColors(coords,p);
  assert(all(coords == square));
  for(int i=0; i < p.length; ++i)
    assert(p[i] == squareColors[i]);
}
EndTest();

StartTest("fitColors: degenerate input");
{
  // One point gives its color everywhere.
  pen p=rgb(1,0.5,0);
  vertexPen vp=fitColors(new triple[] {(1,2,3)},new pen[] {p});
  assert(distance(at(vp,(1,2,3)),p) < fitTolerance);
  assert(distance(at(vp,(10,-2,3)),p) < fitTolerance);

  // Repeated points with one color are no different.
  vertexPen vp=fitColors(array(5,(1,2,3)),array(5,p));
  assert(distance(at(vp,(1,2,3)),p) < fitTolerance);

  // Collinear points are interpolated along their line.
  triple[] line={(0,0,0),(1,1,1),(2,2,2)};
  pen[] lineColors={rgb(1,0,0),rgb(0.5,0,0.5),rgb(0,0,1)};
  assert(fitError(line,lineColors) < fitTolerance);
  assert(fitError(line,lineColors,T) < fitTolerance);
  vertexPen vp=fitColors(line,lineColors);
  assert(distance(at(vp,(0.5,0.5,0.5)),rgb(0.75,0,0.25)) < 0.01);
  // Away from the line the fit is not determined by the data, but it must
  // still be a color.
  real[] c=colors(at(vp,(1,0,0)));
  assert(min(c) >= 0 && max(c) <= 1);
}
EndTest();

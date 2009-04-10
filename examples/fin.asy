import three;
import palette;

int N = 26;
real[] C = array(N,0); 
real[][] A = new real[N][N];
for(int i = 0; i < N; ++i)
  for(int j = 0; j < N; ++j)
    A[i][j] = 0;

real Tb = 100; // deg C
real h = 240; // 240 W/m^2 K
real k = 240; // W/m K
real Tinf = 20; // deg C
real L = 12; // cm
real t = 2; // cm

real delta = 0.01; // 1 cm = 0.01 m

// (1,2)-(2,2)-(3,2)-...-(13,2)
//   |     |     |          |
// (1,1)-(2,1)-(3,1)-...-(13,1)
//
//        |
//       \ /
//        V
//
// 13-14-15-...-24-25
//  |  |  | ...  |  |
//  0- 1- 2-...-11-12

// but, note zero-based array indexing, so counting starts at 0
int indexof(int m, int n)
{
  return 13(n-1)+m-1;
}

int i = 0;

// fixed temperature bottom left
A[i][indexof(1,1)] = 1; C[i] = Tb;
++i;
// fixed temperature middle left
A[i][indexof(1,2)] = 1; C[i] = Tb;
++i;

// interior nodes
for(int m = 2; m<13; ++m)
{
  A[i][indexof(m,2)] = -4;
  A[i][indexof(m-1,2)] = A[i][indexof(m+1,2)] = 1;
  A[i][indexof(m,1)] = 2;
  C[i] = 0;
  ++i;
}

// convective bottom side nodes
for(int m = 2; m<13; ++m)
{
  A[i][indexof(m,1)] = -(2+h*delta/k);
  A[i][indexof(m-1,1)] = A[i][indexof(m+1,1)] = 0.5;
  A[i][indexof(m,2)] = 1;
  C[i] = -h*delta*Tinf/k;
  ++i;
}

// convective bottom right corner node
A[i][indexof(13,2)] = A[i][indexof(12,1)] = 0.5;
A[i][indexof(13,1)] = -(1+h*delta/k);
C[i] = -h*delta*Tinf/k;
++i;

// convective middle right side node
A[i][indexof(13,2)] = -(2+h*delta/k);
A[i][indexof(13,1)] = 1;
A[i][indexof(12,2)] = 1;
C[i] = -h*delta*Tinf/k;
++i;

real[] T = solve(A,C);

pen[] Palette = Gradient(256,blue,cyan,yellow,orange,red);

real[][] T = {T[0:13],T[13:26],T[0:13]};
T = transpose(T);

size3(15cm);
real w = 10;
real h = 5;
currentprojection = orthographic(2*(L,h,w),Y);
draw((L,t,0)--(L,0,0)--(L,0,w)--(0,0,w)--(0,-h,w));
draw((0,t,w)--(0,t+h,w)--(0,t+h,0)--(0,t,0));
draw((L,0,w)--(L,t,w)--(0,t,w)--(0,t,0)--(L,t,0)--(L,t,w));

real wo2 = 0.5*w;
draw((0,0,wo2)--(0,t,wo2)--(L,t,wo2)--(L,0,wo2)--cycle);

// centre points
surface square = surface(shift(-0.5,-0.5,wo2)*unitsquare3);
surface bottomsquare = surface(shift(-0.5,-0.5,wo2)*scale(1,0.5,1)*unitsquare3);
surface topsquare = surface(shift(-0.5,0,wo2)*scale(1,0.5,1)*unitsquare3);
surface leftsquare = surface(shift(-0.5,-0.5,wo2)*scale(0.5,1,1)*unitsquare3);
surface rightsquare = surface(shift(0,-0.5,wo2)*scale(0.5,1,1)*unitsquare3);
surface NEcorner = surface(shift(0,0,wo2)*scale(0.5,0.5,1)*unitsquare3);
surface SEcorner = surface(shift(0,-0.5,wo2)*scale(0.5,0.5,1)*unitsquare3);
surface SWcorner = surface(shift(-0.5,-0.5,wo2)*scale(0.5,0.5,1)*unitsquare3);
surface NWcorner = surface(shift(-0.5,0,wo2)*scale(0.5,0.5,1)*unitsquare3);

material lookupColour(int m,int n)
{
  int index = round(Palette.length*(T[m-1][n-1]-60)/(100-60));
  if(index >= Palette.length) index = Palette.length-1;
  return emissive(Palette[index]);
}

draw(shift(0,1,0)*rightsquare,lookupColour(1,2));
for(int i = 2; i < 13; ++i)
{
  draw(shift(i-1,1,0)*square,lookupColour(i,2));
}
draw(shift(12,1,0)*leftsquare,lookupColour(13,2));

draw(shift(0,2,0)*SEcorner,lookupColour(1,3));
draw(shift(0,0,0)*NEcorner,lookupColour(1,1));
for(int i = 2; i < 13; ++i)
{
  draw(shift(i-1,0,0)*topsquare,lookupColour(i,1));
  draw(shift(i-1,2,0)*bottomsquare,lookupColour(i,3));
}
draw(shift(12,2,0)*SWcorner,lookupColour(13,3));
draw(shift(12,0,0)*NWcorner,lookupColour(13,1));

// annotations
draw("$x$",(0,-h/2,w)--(L/4,-h/2,w),Y,Arrow3(HookHead2(normal=Z)),BeginBar3(Y));
draw("$y$",(0,0,1.05*w)--(0,2t,1.05*w),Z,Arrow3(HookHead2(normal=X)),
     BeginBar3(Z));
draw("$z$",(L,-h/2,0)--(L,-h/2,w/4),Y,Arrow3(HookHead2(normal=X)),BeginBar3(Y));

draw("$L$",(0,-h/4,w)--(L,-h/4,w),-Y,Arrows3(HookHead2(normal=Z)),
     Bars3(Y),PenMargins2);
draw("$w$",(L,-h/4,0)--(L,-h/4,w),-Y,Arrows3(HookHead2(normal=X)),
     Bars3(Y),PenMargins2);
draw("$t$",(1.05*L,0,0)--(1.05*L,t,0),-2Z,Arrows3(HookHead2(normal=Z)),
     Bars3(X),PenMargins2);

label(ZY()*"$T_b$",(0,t+h/2,wo2));

label("$h$,$T_\infty$",(L/2,t+h/2,0),Y);
path3 air = (L/2,t+h/3,w/3.5)--(1.5*L/2,t+2*h/3,w/8);
draw(air,EndArrow3(TeXHead2));
draw(shift(0.5,0,0)*air,EndArrow3(TeXHead2));
draw(shift(1.0,0,0)*air,EndArrow3(TeXHead2));

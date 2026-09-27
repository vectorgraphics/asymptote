import TestLib;
import camera;

bool close(real a, real b)
{
  return abs(a-b) <= 1e-9*max(abs(a),abs(b),1);
}

bool close(pair a, pair b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

triple eye=(6,-8,4), target=(1,1,0);
triple f=unit(target-eye);
triple up=unit(Z-dot(Z,f)*f);
triple right=cross(f,up);

StartTest("camera centers target and spans fov");
{
  projection P=camera(eye,target,fov=60,aspect=4/3);
  real w=abs(P.viewframe[1]), h=abs(P.viewframe[2]);
  assert(close(w/h,4/3));
  assert(close(project(target,P),(w/2,h/2)));
  assert(close(project(eye+rotate(30,up)*f,P),(0,h/2)));
  assert(close(project(eye+rotate(-30,up)*f,P),(w,h/2)));
}
EndTest();

StartTest("camera fovaxis");
{
  projection V=camera(eye,target,fov=40,fovaxis="vertical",aspect=2);
  real w=abs(V.viewframe[1]), h=abs(V.viewframe[2]);
  assert(close(project(eye+rotate(20,right)*f,V),(w/2,h)));
  // For a portrait image, "auto" measures the longer (vertical) side.
  projection A=camera(eye,target,fov=40,aspect=0.5);
  w=abs(A.viewframe[1]);
  h=abs(A.viewframe[2]);
  assert(close(project(eye+rotate(20,right)*f,A),(w/2,h)));
  projection D=camera(eye,target,fov=40,fovaxis="diagonal",aspect=2);
  triple c=D.viewframe[0];
  assert(close(aCos(dot(unit(c-eye),unit(c+D.viewframe[1]+D.viewframe[2]-eye))),
               40));
}
EndTest();

StartTest("camera roll and shift");
{
  projection R=camera(eye,target,roll=90,aspect=1);
  real s=abs(R.viewframe[1]);
  // The image turns counterclockwise: what was up now points left.
  pair z=project(eye+rotate(10,right)*f,R);
  assert(close(z.y,s/2) && z.x < s/2);
  projection S=camera(eye,target,shift=(0.1,0.3),aspect=1);
  s=abs(S.viewframe[1]);
  assert(close(project(target,S),(0.4s,0.2s)));
}
EndTest();

StartTest("camera focus and near");
{
  // By default the picture plane passes through target, and the near
  // clipping plane is much closer to the eye.
  projection P=camera(eye,target);
  assert(close(dot(P.viewframe[0]-eye,f),abs(target-eye)));
  assert(close(P.viewnear,abs(target-eye)/100));
  projection Q=camera(eye,target,focus=2,near=0.5);
  assert(close(dot(Q.viewframe[0]-eye,f),2));
  assert(close(Q.viewnear,0.5));
  // The focus distance scales the picture plane but not the image.
  assert(close(project(target+X,P)/abs(P.viewframe[1]),
               project(target+X,Q)/abs(Q.viewframe[1])));
}
EndTest();

StartTest("lens");
{
  assert(close(lens(18),90));
  assert(close(lens(50,sensor=24),2aTan(0.24)));
}
EndTest();

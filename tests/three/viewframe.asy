import TestLib;
import three;

bool close(pair a, pair b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

bool close(triple a, triple b)
{
  real norm=max(abs(a),abs(b),1);
  return abs(a-b) <= 1e-9*norm;
}

// A skewed picture plane (u and v not perpendicular) seen from an eye well
// off the perpendicular through its center.
triple eye=(1,-2,3);
triple p=(-3,4,-1), u=(5,1,0.5), v=(1.5,0.3,4);

StartTest("viewframe projects onto the picture plane");
{
  projection P=viewframe(eye,p,u,v);
  // A point on the ray from the eye through p+s*u+t*v, on either side of the
  // picture plane, projects to (s*abs(u),t*abs(v)).
  pair[] st={(0,0),(1,0),(0,1),(1,1),(0.3,0.8),(-0.5,1.7)};
  real[] lambda={0.5,1,2.5};
  for(pair z : st)
    for(real l : lambda) {
      triple X=eye+l*(p+z.x*u+z.y*v-eye);
      assert(close(project(X,P),(z.x*abs(u),z.y*abs(v))));
    }
}
EndTest();

StartTest("viewframe modelview is rigid and faces the picture plane");
{
  projection P=viewframe(eye,p,u,v);
  transform3 m=P.T.modelview;
  assert(close(m*eye,O));
  triple x=shiftless(m)*X, y=shiftless(m)*Y, z=shiftless(m)*Z;
  assert(abs(abs(x)-1) < 1e-12 && abs(abs(y)-1) < 1e-12 &&
         abs(abs(z)-1) < 1e-12);
  assert(abs(dot(x,y)) < 1e-12 && abs(dot(y,z)) < 1e-12 &&
         abs(dot(z,x)) < 1e-12);
  // The picture plane is perpendicular to the view axis, in front of the eye,
  // with u along the image x axis.
  triple U=shiftless(m)*u, V=shiftless(m)*v, c=m*p;
  assert(abs(U.z) < 1e-12 && abs(V.z) < 1e-12 && c.z < 0);
  assert(abs(U.y) < 1e-12 && U.x > 0);
}
EndTest();

StartTest("viewframe survives copy and transformation");
{
  projection P=viewframe(eye,p,u,v);
  projection Q=scale3(2)*P.copy();
  assert(Q.absolute);
  assert(Q.viewframe.length == 3);
  assert(close(Q.viewframe[0],p) && close(Q.viewframe[1],u) &&
         close(Q.viewframe[2],v));
  assert(close(project(p+u+v,Q),(abs(u),abs(v))));
}
EndTest();

StartTest("viewframe near clipping distance");
{
  // By default, the near clipping plane is the picture plane.
  projection P=viewframe(eye,p,u,v);
  assert(abs(P.viewnear-dot(eye-p,unit(cross(u,v)))) < 1e-12);
  projection Q=viewframe(eye,p,u,v,near=0.25);
  assert(Q.viewnear == 0.25);
  assert(Q.copy().viewnear == 0.25);
  // The near distance does not affect the projection.
  assert(close(project(p+0.3u+0.6v,Q),project(p+0.3u+0.6v,P)));
}
EndTest();

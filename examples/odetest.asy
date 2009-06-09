import ode;

write("system integration test");
real[] f(real t, real[] x) {
  return new real[] {x[1],1.5*x[0]^2};
}
write(integrate(new real[] {4,-8},f,0,1,100,RK4));
write();

write("simultaneous newton test");
real[] function(real[] x) {
  return new real[] {x[0]^2+x[1]^2-25,(x[0]-6)^2+x[1]^2-25};
}
real[][] fJac(real[] x) {
  return new real[][] {{2*x[0],2*x[1]},{2*(x[0]-6),2*x[1]}};
}
write(newton(function,fJac,new real[] {0,-1}));
write();


write("BVP solver test");
write("Finding initial conditions that solve w''(t)=1.5*w(t), w(0)=4, w(1)=1");
real[] initial(real[] x) {
  return new real[] {4,x[0]};
}

real[] discrepancy(real[] x) {
  write("Error: ",x[0]-1);
  return new real[] {x[0]-1};
}

write(solveBVP(f,0.0,1.0,initial,discrepancy,guess=new real[] {-30},n=10,RK4,
               iterations=10));
write();
write(solveBVP(f,0.0,1.0,initial,discrepancy,guess=new real[] {-30},n=100,RK4,
               iterations=10));
write();
write(solveBVP(f,0.0,1.0,initial,discrepancy,guess=new real[] {-30},n=10000,
               RK4,iterations=10));
write();

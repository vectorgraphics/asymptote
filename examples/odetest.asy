import ode;

write("integration test");
real f(real t, real x) {return cos(x);}
write(integrate(1,f,0,10,0.1,dynamic=true,0.0002,0.0004,RK3BS,verbose=true));
write();

write("system integration test");
real[] f(real t, real[] x) {return new real[] {x[1],1.5*x[0]^2};}
write(integrate(new real[] {4,-8},f,0,1,n=100,dynamic=true,tolmin=0.0002,
                tolmax=0.0004,RK3BS,verbose=false));
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
  real error=x[0]-1;
  write("Error: ",error);
  return new real[] {error};
}

real[] w0=solveBVP(f,0,1,n=10,dynamic=true,tolmin=0.0002,tolmax=0.0004,RK3BS,
                   initial,discrepancy,guess=new real[] {-30},iterations=10);
write(w0);
write();
write(integrate(w0,f,0,1,n=10,dynamic=true,tolmin=0.0002,tolmax=0.0004,RK3BS,
                verbose=false));
write();

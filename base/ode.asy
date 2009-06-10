// for dynamic stepsize adjustment
real stepfactor=2.0;
real minstep=realEpsilon, maxstep=realMax;

struct RKTableau
{
  int order;
  real[] hSteps;
  real[][] weights;
  real[][] finalWeights;

  void operator init(int order, real[] hSteps, real[][] weights,
                     real[][] finalWeights) {
    this.order=order;
    this.hSteps=hSteps;
    this.weights=weights;
    this.finalWeights=finalWeights;
  }
}

RKTableau Euler=RKTableau(1,new real[] {0.0},
                          new real[][] {{}},
                          new real[][] {{1}});

RKTableau RK3=RKTableau(3,new real[] {0,0.5,1},
                        new real[][] {{0.5},{-1,2}},
                        new real[][] {{1/6,2/3,1/6}});

RKTableau RK3=RKTableau(3,new real[] {0,0.5,1},
                        new real[][] {{0.5},{-1,2}},
                        new real[][] {{1/6,2/3,1/6}});

RKTableau RK4=RKTableau(4,new real[] {0,0.5,0.5,1},
                        new real[][] {{0.5},{0,0.5},{0,0,1}},
                        new real[][] {{1/6,1/3,1/3,1/6}});

RKTableau RKCK45=RKTableau(5,new real[] {0,1/5,3/10,3/5,1,7/8},
                           new real[][] {{1/5},
                                         {3/40,9/40},
                                         {3/10,-9/10,6/5},
                                         {-11/54,5/2,-70/27,35/27},
                                         {1631/55296,175/512,575/13824,
                                          44275/110592,253/4096}},
                           new real[][] {{2825/27648,0,18575/48384,13525/55296,
                                          277/14336,1/4}, // 4th order
                                         {37/378,0,250/621,125/594,
                                          0,512/1771}}); // 5th order

RKTableau RKF45=RKTableau(5,new real[] {0,1/4,3/8,12/13,1,1/2},
                          new real[][] {{1/4},
                                        {3/32,9/32},
                                        {1932/2197,-7200/2197,7296/2197},
                                        {439/216,-8,3680/513,-845/4104},
                                        {-8/27,2,-3544/2565,1859/4104,-11/40}},
                          new real[][] {{25/216,0,1408/2565,2197/4104,-1/5,0},
                                        // 4th order
                                        {16/135,0,6656/12825,28561/56430,-9/50,
                                         2/55}}); // 5th order

RKTableau DP45=RKTableau(5,new real[] {0,1/5,3/10,4/5,8/9,1,1},
                         new real[][] {{1/5},
                                       {3/40,9/40},
                                       {44/45,-56/15,32/9},
                                       {19372/6561,-25360/2187,64448/6561,
                                        -212/729},
                                       {9017/3168,-355/33,46732/5247,49/176,
                                        -5103/18656},
                                       {35/384,0,500/1113,125/192,-2187/6784,
                                        11/84}},
                         new real[][] {{35/384,0,500/1113,125/192,-2187/6784,
                                        11/84,0}, // 4th order ? 
                                       {5179/57600,0,7571/16695,393/640,
                                        -92097/339200,187/2100,1/40}});
// 5th order ? 

triple RKstep(real t, real y, real f(real t, real y), real h,
              bool dynamic, real tolmin, real tolmax, RKTableau tableau)
{
  dynamic=(dynamic && tableau.finalWeights.length > 1);
  real[] samplePositions=t+h*tableau.hSteps;
  real[] predictions=new real[] {f(samplePositions[0],y)};
  for(int i=1; i < tableau.hSteps.length; ++i)
    predictions.push(f(samplePositions[i],
                       y+h*dot(tableau.weights[i-1],predictions)));

  real[] yIncrement=tableau.finalWeights*(h*predictions);

  real tinc=0, yinc=0;
  if(dynamic) {
    // check difference between highest order estimates
    real errorEstimate =
      abs(yIncrement[yIncrement.length-1]-
          yIncrement[yIncrement.length-2]);
    if(errorEstimate > tolmax)
      h=max(h/stepfactor,minstep);
    else {
      if(errorEstimate < tolmin)
        h=min(h*stepfactor,maxstep);
      tinc=h;
      yinc=yIncrement[0];
    }
  } else {
    tinc=h;
    yinc=yIncrement[yIncrement.length-1]; // step forward with highest order
  }
  return(tinc,yinc,h);
}

// integrate dy/dt=f(t,y) from a to b using initial conditions y,
// specifying either n steps or the step size h.
real integrate(real y, real f(real t, real y), real a, real b=a, int n=0,
               real h=0, bool dynamic=false, real tolmin=0, real tolmax=0,
               RKTableau tableau)
{
  if(h == 0) {
    if(b == a) return y;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }
  real t=a;
  while(abs(t-b) > abs(h)) {
    triple p=RKstep(t,y,f,h,dynamic,tolmin,tolmax,tableau);
    t += p.x;
    y += p.y;
    h=p.z;
  }
  h=b-t;
  while(abs(t-b) > realEpsilon*abs(b)) {
    triple p=RKstep(t,y,f,h,dynamic,tolmin,tolmax,tableau);
    t += p.x;
    y += p.y;
    h=p.z;
  }
  return y;
}

// integrate a set of equations, dy/dt=f(t,y), from a to b using initial
// conditions y, specifying either n steps or the step size h.
real[] integrate(real[] y, real[] f(real, real[]), real a, real b=a, int n=0,
                 real h=0, RKTableau tableau)
{
  if(h == 0) {
    if(b == a) return y;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }
  real[] y=copy(y);
  real t=a;
  int m=tableau.hSteps.length;
  
  for(int i=0; i < n; ++i) {
    real[] samplePositions=t+h*tableau.hSteps;
    real[][] derivatives={f(t,y)};
    for(int i=1; i < tableau.hSteps.length; ++i)
      derivatives.push(f(samplePositions[i],
                         y+h*tableau.weights[i-1]*derivatives));

    y += h*tableau.finalWeights[tableau.finalWeights.length-1]*derivatives ;
    t += h;
  }
  return y;
}

real[][] finiteDifferenceJacobian(real[] f(real[]), real[] t)
{
  real[] ft=f(t);

  real h=sqrtEpsilon*abs(ft[0]);
  if(h < sqrtEpsilon) h=sqrtEpsilon;

  real[][] J=new real[t.length][ft.length];
  real[] ti=copy(t);
  real tlast=ti[0];
  ti[0] += h;
  J[0]=(f(ti)-ft)/h;
  for(int i = 1; i < t.length; ++i) {
    ti[i-1]=tlast;
    tlast=ti[i];
    ti[i] += h;
    J[i]=(f(ti)-ft)/h;
  }
  return transpose(J);
}

// solve simultaneous nonlinear system by newton's method
real[] newton(int iterations=100, real[] f(real[]), real[][] jacobian(real[]),
              real[] t)
{
  real[] t=copy(t);
  for(int i=0; i < iterations; ++i)
    t += solve(jacobian(t),-f(t));
  return t;
}

real[] solveBVP(real[] f(real, real[]), real a, real b,
                real[] initial(real[]), real[] discrepancy(real[]),
                real[] guess, int n, RKTableau tableau, int iterations=100)
{
  real[] g(real[] t) {
    return discrepancy(integrate(initial(t),f,a,b,n,tableau));
  }
  real[][] jacobian(real[] t) {return finiteDifferenceJacobian(g,t);}
  return initial(newton(iterations,g,jacobian,guess));
}

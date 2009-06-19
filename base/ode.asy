real stepfactor=2.0; // Maximum dynamic step size adjustment factor.

struct coefficients
{
  real[] factors;
  real[][] weights;
  real[] highOrderWeights;
  real[] lowOrderWeights;
}

struct RKTableau
{
  int order;
  real[] steps;
  coefficients a;
  void stepDependence(real h, real c, coefficients a) {}
   
  real pgrow;
  real pshrink;
  bool exponential;

  void operator init(int order, real[][] weights, real[] highOrderWeights,
                     real[] lowOrderWeights=new real[],
                     real[] steps=sequence(new real(int i) {
                         return sum(weights[i]);},weights.length),
                     void stepDependence(real, real, coefficients)=null) {
    this.order=order;
    this.steps=steps;
    a.factors=array(steps.length+1,1);
    a.weights=weights;
    a.highOrderWeights=highOrderWeights;
    a.lowOrderWeights=lowOrderWeights;
    if(stepDependence != null) {
      this.stepDependence=stepDependence;
      exponential=true;
    }
    pgrow=(order > 0) ? 1/order : 0;
    pshrink=(order > 1) ? 1/(order-1) : pgrow;
  }
}

// First-Order Euler
RKTableau Euler=RKTableau(1,new real[][],
                          new real[] {1});

RKTableau E_Euler=RKTableau(1,new real[][], new real[] {1},
                            new void (real h, real c, coefficients a) {
                              real x=-c*h;
                              a.factors[0]=exp(x);
                              a.highOrderWeights[0]=x != 0 ? expm1(x)/x : 1;
                            });

// Second-Order Runge-Kutta
RKTableau RK2=RKTableau(2,new real[][] {{1/2}},
                        new real[] {0,1});

// Second-Order Predictor-Corrector
RKTableau PC=RKTableau(2,new real[][] {{1}},
                       new real[] {1/2,1/2});

// Third-Order Classical Runge-Kutta
RKTableau RK3=RKTableau(3,new real[][] {{1/2},{-1,2}},
                        new real[] {1/6,2/3,1/6});

// Third-Order Bogacki-Shampine Runge-Kutta
RKTableau RK3BS=RKTableau(3,new real[][] {{1/2},{0,3/4}},
                          new real[] {2/9,1/3,4/9}, // 3rd order
                          new real[] {7/24,1/4,1/3,1/8}); // 2nd order

// Fourth-Order Classical Runge-Kutta
RKTableau RK4=RKTableau(4,new real[][] {{1/2},{0,1/2},{0,0,1}},
                        new real[] {1/6,1/3,1/3,1/6});

// Fifth-Order Cash-Karp Runge-Kutta
RKTableau RK5=RKTableau(5,new real[][] {{1/5},
                                        {3/40,9/40},
                                        {3/10,-9/10,6/5},
                                        {-11/54,5/2,-70/27,35/27},
                                        {1631/55296,175/512,575/13824,
                                         44275/110592,253/4096}},
  new real[] {37/378,0,250/621,125/594,
              0,512/1771},  // 5th order
  new real[] {2825/27648,0,18575/48384,13525/55296,
              277/14336,1/4}); // 4th order

// Fifth-Order Fehlberg Runge-Kutta
RKTableau RK5F=RKTableau(5,new real[][] {{1/4},
                                         {3/32,9/32},
                                         {1932/2197,-7200/2197,7296/2197},
                                         {439/216,-8,3680/513,-845/4104},
                                         {-8/27,2,-3544/2565,1859/4104,
                                          -11/40}},
  new real[] {16/135,0,6656/12825,28561/56430,-9/50,2/55}, // 5th order
  new real[] {25/216,0,1408/2565,2197/4104,-1/5,0}); // 4th order

// Fifth-Order Dormand-Prince Runge-Kutta
RKTableau RK5DP=RKTableau(5,new real[][] {{1/5},
                                          {3/40,9/40},
                                          {44/45,-56/15,32/9},
                                          {19372/6561,-25360/2187,64448/6561,
                                           -212/729},
                                          {9017/3168,-355/33,46732/5247,49/176,
                                           -5103/18656}},
  new real[] {35/384,0,500/1113,125/192,-2187/6784,
              11/84}, // 5th order
  new real[] {5179/57600,0,7571/16695,393/640,
              -92097/339200,187/2100,1/40}); // 4th order

real error(real error, real initial, real lowOrder, real norm, real diff) 
{
  if(initial != 0.0 && lowOrder != initial) {
    static real epsilon=realMin/realEpsilon;
    real denom=max(abs(norm),abs(initial))+epsilon;
    return max(error,max(abs(diff)/denom));
  }
  return error;
}

real adjust(real h, real error, real t, real c, real tolmin, real tolmax,
            real dtmin, real dtmax, RKTableau tableau, bool verbose=true) 
{
  real dt=h;
  void report(real t) {
    if(h != dt) {
      tableau.stepDependence(h,c,tableau.a);
      if(verbose)
        write("Time step changed from "+(string) dt+" to "+(string) h+" at t="+
              (string) t+".");
    }
  }
  if(error > tolmax) {
    h=max(h*max((tolmin/error)^tableau.pshrink,1/stepfactor),dtmin);
    report(t);
    return h;
  }
  if(error > 0 && error < tolmin) {
    h=min(h*min((tolmin/error)^tableau.pgrow,stepfactor),dtmax);
    report(t+dt);
  }
  return h;
}

// Integrate dy/dt+cy=f(t,y) from a to b using initial conditions y,
// specifying either the step size h or the number of steps n.
real integrate(real y, real c=0, real g(real t, real y), real a, real b=a,
               real h=0, int n=0, bool dynamic=false, real tolmin=0,
               real tolmax=0, real dtmin=0, real dtmax=realMax,
               RKTableau tableau, bool verbose=false)
{
  real f(real t, real y)=(c == 0 || tableau.exponential) ? g :
    new real(real t, real y) {return g(t,y)-c*y;};

  if(h == 0) {
    if(b == a) return y;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }

  tableau.stepDependence(h,c,tableau.a);
      
  real t=a;
  real f0;
  if(tableau.a.lowOrderWeights.length == 0) dynamic=false;
  bool fsal=dynamic &&
    (tableau.a.lowOrderWeights.length > tableau.a.highOrderWeights.length);
  if(fsal) f0=f(t,y);

  while(t < b) {
    real[] predictions={fsal ? f0 : f(t,y)};
    for(int i=0; i < tableau.steps.length; ++i)
      predictions.push(f(t+h*tableau.steps[i],
                         tableau.a.factors[i]*y+h*dot(tableau.a.weights[i],
                                                      predictions)));

    real highOrder=h*dot(tableau.a.highOrderWeights,predictions);
    real Y=tableau.a.factors[tableau.steps.length]*y;
    if(dynamic) {
      real f1;
      if(fsal) {
        f1=f(t+h,Y+highOrder);
        predictions.push(f1);
      }
      real lowOrder=h*dot(tableau.a.lowOrderWeights,predictions);
      real error;
      error=error(error,y,Y+lowOrder,Y+highOrder,highOrder-lowOrder);
      real dt=h;
      h=adjust(h,error,t,c,tolmin,tolmax,dtmin,min(dtmax,b-t-h),tableau,verbose);
      if(h >= dt) {
        t += dt;
        y=Y+highOrder;
        f0=f1;
      }
    } else {
      t += h;
      y=Y+highOrder;
    }
    real remain=b-t;
    if(h > remain) {
      h=remain;
      tableau.stepDependence(h,c,tableau.a);
    }
    if(t >= b || t+h == t) break;
  }
  return y;
}

// Integrate a set of equations, dy/dt=f(t,y), from a to b using initial
// conditions y, specifying either the step size h or the number of steps n.
real[] integrate(real[] y, real[] f(real t, real[] y), real a, real b=a,
                 real h=0, int n=0, bool dynamic=false,
                 real tolmin=0, real tolmax=0, real dtmin=0, real dtmax=realMax,
                 RKTableau tableau, bool verbose=false)
{
  if(h == 0) {
    if(b == a) return y;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }
  real[] y=copy(y);
  real t=a;
  real[] f0;
  if(tableau.a.lowOrderWeights.length == 0) dynamic=false;
  bool fsal=dynamic &&
    (tableau.a.lowOrderWeights.length > tableau.a.highOrderWeights.length);
  if(fsal) f0=f(t,y);

  while(t < b) {
    real[][] predictions={fsal ? f0 : f(t,y)};
    for(int i=0; i < tableau.steps.length; ++i)
      predictions.push(f(t+h*tableau.steps[i],
                         y+h*tableau.a.weights[i]*predictions));

    real[] highOrder=h*tableau.a.highOrderWeights*predictions;
    if(dynamic) {
      real[] f1;
      if(fsal) {
        f1=f(t+h,y+highOrder);
        predictions.push(f1);
      }
      real[] lowOrder=h*tableau.a.lowOrderWeights*predictions;
      real error;
      for(int i=0; i < y.length; ++i)
        error=error(error,y[i],y[i]+lowOrder[i],y[i]+highOrder[i],
                    highOrder[i]-lowOrder[i]);
      real dt=h;
      h=adjust(h,error,t,0,tolmin,tolmax,dtmin,min(dtmax,b-t-h),tableau,verbose);
      if(h >= dt) {
        t += dt;
        y += highOrder;
        f0=f1;
      }
    } else {
      t += h;
      y += highOrder;
    }
    h=min(h,b-t);
    if(t >= b || t+h == t) break;
  }
  return y;
}

real[][] finiteDifferenceJacobian(real[] f(real[]), real[] t,
                                  real[] h=sqrtEpsilon*abs(t))
{
  real[] ft=f(t);
  real[][] J=new real[t.length][ft.length];
  real[] ti=copy(t);
  real tlast=ti[0];
  ti[0] += h[0];
  J[0]=(f(ti)-ft)/h[0];
  for(int i=1; i < t.length; ++i) {
    ti[i-1]=tlast;
    tlast=ti[i];
    ti[i] += h[i];
    J[i]=(f(ti)-ft)/h[i];
  }
  return transpose(J);
}

// Solve simultaneous nonlinear system by Newton's method.
real[] newton(int iterations=100, real[] f(real[]), real[][] jacobian(real[]),
              real[] t)
{
  real[] t=copy(t);
  for(int i=0; i < iterations; ++i)
    t += solve(jacobian(t),-f(t));
  return t;
}

real[] solveBVP(real[] f(real, real[]), real a, real b=a, real h=0, int n=0,
                real[] initial(real[]), real[] discrepancy(real[]),
                real[] guess, RKTableau tableau, int iterations=100)
{
  real[] g(real[] t) {
    return discrepancy(integrate(initial(t),f,a,b,h,n,tableau));
  }
  real[][] jacobian(real[] t) {return finiteDifferenceJacobian(g,t);}
  return initial(newton(iterations,g,jacobian,guess));
}

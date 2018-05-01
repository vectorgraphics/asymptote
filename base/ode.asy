real stepfactor=2; // Maximum dynamic step size adjustment factor.

struct coefficients
{
  real[] steps;
  real[] factors;
  real[][] weights;
  real[] highOrderWeights;
  real[] lowOrderWeights;
}

struct RKTableau
{
  int order;
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
    a.steps=steps;
    a.factors=array(a.steps.length+1,1);
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

real[] Coeff={1,1/2,1/6,1/24,1/120,1/720,1/5040,1/40320,1/362880,1/3628800,
              1/39916800.0,1/479001600.0,1/6227020800.0,1/87178291200.0,
              1/1307674368000.0,1/20922789888000.0,1/355687428096000.0,
              1/6402373705728000.0,1/121645100408832000.0,
              1/2432902008176640000.0,1/51090942171709440000.0,
              1/1124000727777607680000.0};

real phi1(real x) {return x != 0 ? expm1(x)/x : 1;}

real phi2(real x)
{
  real x2=x*x;
  if(fabs(x) > 1) return (exp(x)-x-1)/x2;
  real x3=x2*x;
  real x5=x2*x3;
  if(fabs(x) < 0.1) 
    return Coeff[1]+x*Coeff[2]+x2*Coeff[3]+x3*Coeff[4]+x2*x2*Coeff[5]
      +x5*Coeff[6]+x3*x3*Coeff[7]+x5*x2*Coeff[8]+x5*x3*Coeff[9];
    else {
      real x7=x5*x2;
      real x8=x7*x;
      return Coeff[1]+x*Coeff[2]+x2*Coeff[3]+x3*Coeff[4]+x2*x2*Coeff[5]
        +x5*Coeff[6]+x3*x3*Coeff[7]+x7*Coeff[8]+x8*Coeff[9]
        +x8*x*Coeff[10]+x5*x5*Coeff[11]+x8*x3*Coeff[12]+x7*x5*Coeff[13]+
        x8*x5*Coeff[14]+x7*x7*Coeff[15]+x8*x7*Coeff[16]+x8*x8*Coeff[17];
    }
}

real phi3(real x)
{
  real x2=x*x;
  real x3=x2*x;
  if(fabs(x) > 1.6) return (exp(x)-0.5*x2-x-1)/x3;
  real x5=x2*x3;
  if(fabs(x) < 0.1) 
    return Coeff[2]+x*Coeff[3]+x2*Coeff[4]+x3*Coeff[5]
      +x2*x2*Coeff[6]+x5*Coeff[7]+x3*x3*Coeff[8]+x5*x2*Coeff[9]
      +x5*x3*Coeff[10];
    else {
      real x7=x5*x2;
      real x8=x7*x;
      real x16=x8*x8;
      return Coeff[2]+x*Coeff[3]+x2*Coeff[4]+x3*Coeff[5]
        +x2*x2*Coeff[6]+x5*Coeff[7]+x3*x3*Coeff[8]+x5*x2*Coeff[9]
        +x5*x3*Coeff[10]+x8*x*Coeff[11]
        +x5*x5*Coeff[12]+x8*x3*Coeff[13]+x7*x5*Coeff[14]
        +x8*x5*Coeff[15]+x7*x7*Coeff[16]+x8*x7*Coeff[17]+x16*Coeff[18]
        +x16*x*Coeff[19]+x16*x2*Coeff[20];
    }
}

void expfactors(real x, coefficients a) 
{
  for(int i=0; i < a.steps.length; ++i)
    a.factors[i]=exp(x*a.steps[i]);
  a.factors[a.steps.length]=exp(x);
}
      
// First-Order Euler
RKTableau Euler=RKTableau(1,new real[][], new real[] {1});

// First-Order Exponential Euler
RKTableau E_Euler=RKTableau(1,new real[][], new real[] {1},
                            new void(real h, real c, coefficients a) {
                              real x=-c*h;
                              expfactors(x,a);
                              a.highOrderWeights[0]=phi1(x);
                            });

// Second-Order Runge-Kutta
RKTableau RK2=RKTableau(2,new real[][] {{1/2}},
                        new real[] {0,1}, // 2nd order
                        new real[] {1,0}); // 1st order

// Second-Order Exponential Runge-Kutta
RKTableau E_RK2=RKTableau(2,new real[][] {{1/2}},
                          new real[] {0,1}, // 2nd order
                          new real[] {1,0}, // 1st order
                          new void(real h, real c, coefficients a) {
                            real x=-c*h;
                            expfactors(x,a);
                            a.weights[0][0]=1/2*phi1(x/2);
                            real w=phi1(x);
                            a.highOrderWeights[0]=0;
                            a.highOrderWeights[1]=w;
                            a.lowOrderWeights[0]=w;
                          });

// Second-Order Predictor-Corrector
RKTableau PC=RKTableau(2,new real[][] {{1}},
                       new real[] {1/2,1/2}, // 2nd order
                       new real[] {1,0}); // 1st order

// Second-Order Exponential Predictor-Corrector
RKTableau E_PC=RKTableau(2,new real[][] {{1}},
                         new real[] {1/2,1/2}, // 2nd order
                         new real[] {1,0}, // 1st order
                         new void(real h, real c, coefficients a) {
                           real x=-c*h;
                           expfactors(x,a);
                           real w=phi1(x);
                           a.weights[0][0]=w;
                           a.highOrderWeights[0]=w/2;
                           a.highOrderWeights[1]=w/2;
                           a.lowOrderWeights[0]=w;
                         });

// Third-Order Classical Runge-Kutta
RKTableau RK3=RKTableau(3,new real[][] {{1/2},{-1,2}},
                        new real[] {1/6,2/3,1/6});

// Third-Order Bogacki-Shampine Runge-Kutta
RKTableau RK3BS=RKTableau(3,new real[][] {{1/2},{0,3/4}},
                          new real[] {2/9,1/3,4/9}, // 3rd order
                          new real[] {7/24,1/4,1/3,1/8}); // 2nd order

// Third-Order Exponential Bogacki-Shampine Runge-Kutta
RKTableau E_RK3BS=RKTableau(3,new real[][] {{1/2},{0,3/4}},
                            new real[] {2/9,1/3,4/9}, // 3rd order
                            new real[] {7/24,1/4,1/3,1/8}, // 2nd order
                            new void(real h, real c, coefficients a) {
                              real x=-c*h;
                              expfactors(x,a);
                              real w=phi1(x);
                              real w2=phi2(x);
                              a.weights[0][0]=1/2*phi1(x/2);
                              real a11=9/8*phi2(3/4*x)+3/8*phi2(x/2);
                              a.weights[1][0]=3/4*phi1(3/4*x)-a11;
                              a.weights[1][1]=a11;
                              real a21=1/3*w;
                              real a22=4/3*w2-2/9*w;
                              a.highOrderWeights[0]=w-a21-a22;
                              a.highOrderWeights[1]=a21;
                              a.highOrderWeights[2]=a22;
                              a.lowOrderWeights[0]=w-17/12*w2;
                              a.lowOrderWeights[1]=w2/2;
                              a.lowOrderWeights[2]=2/3*w2;
                              a.lowOrderWeights[3]=w2/4;
                            });

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
  if(initial != 0 && lowOrder != initial) {
    static real epsilon=realMin/realEpsilon;
    real denom=max(abs(norm),abs(initial))+epsilon;
    return max(error,max(abs(diff)/denom));
  }
  return error;
}

void report(real old, real h, real t)
{
  write("Time step changed from "+(string) old+" to "+(string) h+" at t="+
        (string) t+".");
}

real adjust(real h, real error, real tolmin, real tolmax, RKTableau tableau)
{
  if(error > tolmax)
    h *= max((tolmin/error)^tableau.pshrink,1/stepfactor);
  else if(error > 0 && error < tolmin)
    h *= min((tolmin/error)^tableau.pgrow,stepfactor);
  return h;
}

struct solution
{
  real[] t;
  real[] y;
}

void write(solution S) 
{
  for(int i=0; i < S.t.length; ++i)
    write(S.t[i],S.y[i]);
}

// Integrate dy/dt+cy=f(t,y) from a to b using initial conditions y,
// specifying either the step size h or the number of steps n.
solution integrate(real y, real c=0, real f(real t, real y), real a, real b=a,
                   real h=0, int n=0, bool dynamic=false, real tolmin=0,
                   real tolmax=0, real dtmin=0, real dtmax=realMax,
                   RKTableau tableau, bool verbose=false)
{
  solution S;
  S.t=new real[] {a};
  S.y=new real[] {y};

  if(h == 0) {
    if(b == a) return S;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }

  real F(real t, real y)=(c == 0 || tableau.exponential) ? f :
    new real(real t, real y) {return f(t,y)-c*y;};

  tableau.stepDependence(h,c,tableau.a);
      
  real t=a;
  real f0;
  if(tableau.a.lowOrderWeights.length == 0) dynamic=false;
  bool fsal=dynamic &&
    (tableau.a.lowOrderWeights.length > tableau.a.highOrderWeights.length);
  if(fsal) f0=F(t,y);

  real dt=h;
  while(t < b) {
    h=min(h,b-t);
    if(t+h == t) break;
    if(h != dt) {
      if(verbose) report(dt,h,t);
      tableau.stepDependence(h,c,tableau.a);
      dt=h;
    }
 
    real[] predictions={fsal ? f0 : F(t,y)};
    for(int i=0; i < tableau.a.steps.length; ++i)
      predictions.push(F(t+h*tableau.a.steps[i],
                         tableau.a.factors[i]*y+h*dot(tableau.a.weights[i],
                                                      predictions)));

    real highOrder=h*dot(tableau.a.highOrderWeights,predictions);
    real y0=tableau.a.factors[tableau.a.steps.length]*y;
    if(dynamic) {
      real f1;
      if(fsal) {
        f1=F(t+h,y0+highOrder);
        predictions.push(f1);
      }
      real lowOrder=h*dot(tableau.a.lowOrderWeights,predictions);
      real error;
      error=error(error,y,y0+lowOrder,y0+highOrder,highOrder-lowOrder);
      h=adjust(h,error,tolmin,tolmax,tableau);
      if(h >= dt) {
        t += dt;
        y=y0+highOrder;
        S.t.push(t);
        S.y.push(y);
        f0=f1;
      }
      h=min(max(h,dtmin),dtmax);
    } else {
      t += h;
      y=y0+highOrder;
      S.t.push(t);
      S.y.push(y);
    }
  }
  return S;
}

struct Solution
{
  real[] t;
  real[][] y;
}

void write(Solution S) 
{
  for(int i=0; i < S.t.length; ++i) {
    write(S.t[i],tab);
    for(real y : S.y[i])
      write(y,tab);
    write();
  }
}

// Integrate a set of equations, dy/dt=f(t,y), from a to b using initial
// conditions y, specifying either the step size h or the number of steps n.
Solution integrate(real[] y, real[] f(real t, real[] y), real a, real b=a,
                   real h=0, int n=0, bool dynamic=false,
                   real tolmin=0, real tolmax=0, real dtmin=0,
                   real dtmax=realMax, RKTableau tableau, bool verbose=false)
{
  Solution S;
  S.t=new real[] {a};
  S.y=new real[][] {copy(y)};
      
  if(h == 0) {
    if(b == a) return S;
    if(n == 0) abort("Either n or h must be specified");
    else h=(b-a)/n;
  }
  real t=a;
  real[] f0;
  if(tableau.a.lowOrderWeights.length == 0) dynamic=false;
  bool fsal=dynamic &&
    (tableau.a.lowOrderWeights.length > tableau.a.highOrderWeights.length);
  if(fsal) f0=f(t,y);

  real dt=h;
  while(t < b) {
    h=min(h,b-t);
    if(t+h == t) break;
    if(h != dt) {
      if(verbose) report(dt,h,t);
      dt=h;
    }

    real[][] predictions={fsal ? f0 : f(t,y)};
    for(int i=0; i < tableau.a.steps.length; ++i)
      predictions.push(f(t+h*tableau.a.steps[i],
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
      h=adjust(h,error,tolmin,tolmax,tableau);
      if(h >= dt) {
        t += dt;
        y += highOrder;
        S.t.push(t);
        S.y.push(y);
        f0=f1;
      }
      h=min(max(h,dtmin),dtmax);
    } else {
      t += h;
      y += highOrder;
      S.t.push(t);
      S.y.push(y);
    }
  }
  return S;
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
                bool dynamic=false, real tolmin=0, real tolmax=0, real dtmin=0,
                real dtmax=realMax, RKTableau tableau, bool verbose=false,
                real[] initial(real[]), real[] discrepancy(real[]),
                real[] guess, int iterations=100)
{
  real[] g(real[] t) {
    real[][] y=integrate(initial(t),f,a,b,h,n,dynamic,tolmin,tolmax,dtmin,dtmax,
                         tableau,verbose).y;return discrepancy(y[y.length-1]);
  }
  real[][] jacobian(real[] t) {return finiteDifferenceJacobian(g,t);}
  return initial(newton(iterations,g,jacobian,guess));
}

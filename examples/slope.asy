import ode;
import graph;
import math;
size(200,200,IgnoreAspect);

real f(real t, real y) {return cos(y);}
//real f(real t, real y) {return 1/(1+y);}
typedef real function(real,real);

real a=0;
real b=1;
real y0=0;

real L[]={1,2};

int M=L.length; // Number of modes.

//real Y0[]=array(M,y0);
real Y0[]=new real[] {-1,2};

real[] F(real t, real[] y) {
  return sequence(new real(int m) {return f(t,y[M-m-1]);},M);
    //  return new real[] {exp((L[1]-1)*t)*y[1],
    //      -exp(-(L[1]-1)*t)*y[0]};
  //  return new real[]{-y[0]^2};
}

real[] G(real t, real[] y) {
  return F(t,y)-sequence(new real(int m) {return L[m]*y[m];},M);
}

real lambda=sqrt(0.5);
real[] tau,error,error2;
int n=25;

real order=3;

for(int i=0; i < n-1; ++i) {
  real dt=(b-a)*lambda^(n-i);
  Solution S=integrate(Y0,L,F,a,b,dt,dynamic=false,0.0002,0.0004,ERK3BS,verbose=false);
  real maxnorm=0;

  Solution E=integrate(Y0,G,a,b,1e-2*dt,dynamic=false,0.0002,0.0004,RK5);
  real[] exact=E.y[E.y.length-1];

  //  real[] exact=new real[] {exp(-b)*sin(b),exp(-L[1]*b)*cos(b)};
  for(int m=0; m < M; ++m)
    maxnorm=max(maxnorm,abs(S.y[S.y.length-1][m]-exact[m]));
    if(maxnorm != 0) {
      tau.push(dt);
      //      error.push(dt^-(order+1)*maxnorm);
            error.push(maxnorm);
    }
}

/*
for(int i=0; i < n-1; ++i) {
  real dt=(b-a)*lambda^(n-i);
  real maxnorm=0;
  for(int m=0; m < M; ++m) {
    solution S=integrate(Y0[m],L[m],f,a,b,dt,dynamic=false,0.000,1000,RK4_375,verbose=false);
    maxnorm=max(maxnorm,abs(S.y[S.y.length-1]-exact[m]));
  }
  error2.push(dt^-order*maxnorm);
}
*/

//scale(Log,Log);
scale(Log,Linear);

//draw(graph(tau,error),marker(scale(0.8mm)*unitcircle,red));
//draw(graph(tau,error2),marker(scale(0.8mm)*unitcircle,blue));

int[] index=sequence(error.length-1);
real[] slope=log(error[index+1]/error[index])/log(tau[index+1]/tau[index]);
real[] t=sqrt(tau[index]*tau[index+1]);
//write(t,slope);
draw(graph(t,slope),red);



xaxis("$\tau$",BottomTop,LeftTicks);
yaxis("$e/\tau^"+string(order)+"$",LeftRight,RightTicks);

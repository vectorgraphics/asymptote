import math;
import featpost3D;

texpreamble("\renewcommand\vec[1]{{\bf{#1}}}");

path quartcirc(vector CenterPos, vector A, vector B) {
  // A generalized version will be developed later
  real G;
  vector vec[], Dna, Dnb, diff;
  guide al;
  
  vector AngulMom = -ncrossprod(A-CenterPos, B-CenterPos);
  real Radius = length(A-CenterPos);
  
  vec[1] = (A-CenterPos)/Radius;
  for (int ind = 2; ind <= 8;  ind += 2) {
    vec[ind+1] = ncrossprod( vec[ind-1], AngulMom );
    vec[ind] = N( vec[ind-1] + vec[ind+1] ); 
  }
  G = conorm( Radius*( vec[1] - vec[2] ) )/3;
  al = rp(Radius*vec[1]+CenterPos);
  for (int ind = 2; ind <= 3; ind += 1) {
    Dna = ncrossprod(ncrossprod(vec[ind-1],vec[ind]),vec[ind-1]);
    Dnb = ncrossprod(ncrossprod(vec[ind],vec[ind-1]),vec[ind]);
    al = al..controls rp(Radius*vec[ind-1]+CenterPos+G*Dna) 
             and rp(Radius*vec[ind]+CenterPos+G*Dnb)
           ..rp(Radius*vec[ind]+CenterPos);
  }
  return (path)al;
}

vector pO,pB;
real radius, aux, lambda;
pen thickp;

f = vector(4,1,2); 
radius = 0.8;
thickp = linewidth(0.5mm);

filldraw(rp(vector(1.2,0,0))--rp(vector(1.2,0,1.2))--
         rp(vector(0,0,1.2))--rp(vector(0,0,0))--cycle,
         gray(.9), gray(.9));
filldraw(rp(vector(0,1.2,0))--rp(vector(0,1.2,1.2))--
         rp(vector(0,0,1.2))--rp(vector(0,0,0))--cycle,
         gray(.9), gray(.9));
filldraw(rp(vector(1.2,0,0))--rp(vector(1.2,1.2,0))--
         rp(vector(0,1.2,0))--rp(vector(0,0,0))--cycle,
         gray(0.9), gray(.9));

cartaxes(1.5,1.5,1.5);    

aux = 60;
lambda = 37;
pO = vector(Cos(aux)*radius,Sin(aux)*radius,0);
draw(rp(vector(0,0,0))--rp(pO));
pB = vector(Cos(aux)*radius*Sin(lambda),
            Sin(aux)*radius*Sin(lambda),
            Cos(lambda)*radius);
draw(rp(vector(0,0,0))--rp(pB),Arrow,PenMargin);
 
real sirk_rad;
sirk_rad = sqrt(pB.x^2+pB.y^2);
draw(quartcirc(vector(0,0,pB.z), 
         vector(sirk_rad,0,pB.z), 
         vector(0,sirk_rad,pB.z)));
draw(quartcirc(vector(0,0,0),
         vector(0,0,radius),
         vector(Cos(aux)*radius,Sin(aux)*radius,0)));

angline(vector(0,0,pB.z), pB, vector(0,0,0), 0.15, "$\lambda$", N+0.2*E);
label("$\vec{Q}$", rp(pB),N);
label("$\rm O$", rp(vector(0,0,0)),W);
label("$\vec{R}$",interp(rp(vector(0,0,0)),rp(pB),0.7),S);

draw(quartcirc(vector(0,0,0),
     radius*vector(0,0,1),
     radius*vector(1,0,0)), thickp);
draw(quartcirc(vector(0,0,0),
     radius*vector(0,0,1),
     radius*vector(0,1,0)), thickp);
draw(quartcirc(vector(0,0,0),
     radius*vector(1,0,0),
     radius*vector(0,1,0)), thickp);


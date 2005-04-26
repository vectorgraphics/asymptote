import math;
import featpost3D;

texpreamble("\usepackage{bm}");
texpreamble("\renewcommand\vec[1]{{\bm{#1}}}");

vector point_on_sphere(vector CenterPos, real rho, real theta, real phi)
{
  return CenterPos+
    vector(Cos(phi)*rho*Sin(theta), Sin(phi)*rho*Sin(theta), Cos(theta)*rho);
}

path circseg(vector CenterPos, real rho, pair r_theta, pair r_phi, int n= 4) {
  /* Draws circle segments (rigorouscircle used as base)
    ====================================================
    CenterPos - the center of the sphere
    rho - the radius of the sphere
    r_theta=(initial theta, end theta)
    r_phi=(initial phi, end phi)
    n - the number of interpolation points
    NOTE : keep either theta or phi constant.
  */
  vector curr, prev, Dna, Dnb, vec1, vec2;
  real theta_step = (r_theta.y-r_theta.x)/n,
       phi_step = (r_phi.y-r_phi.x)/n,
       theta = r_theta.x,
       phi = r_phi.x,
       scl;
  guide G;
  
  curr = point_on_sphere(CenterPos,rho,theta,phi);
  G = rp(curr);
  vec2 = (curr-CenterPos)/rho;

  for (int i = 1; i <= n; i += 1) {
    prev = curr;          vec1 = vec2;
    theta += theta_step;  phi += phi_step;
    curr = point_on_sphere(CenterPos,rho,theta,phi);
    vec2 = (curr-CenterPos)/rho;
    // inefficient now, but necessary if rho is made variable as well.
    scl = length( rho*(vec1 - vec2) )/3;
    Dna = ncrossprod(ncrossprod(vec1,vec2),vec1);
    Dnb = ncrossprod(ncrossprod(vec2,vec1),vec2);
    G = G .. controls rp(prev+scl*Dna) and rp(curr+scl*Dnb) .. rp(curr);
  }
  return (path)G;
}

pen thickp = linewidth(0.5mm);
real radius = 0.8,
     lambda = 37,
     aux = 60;
vector pO,pB;

f = vector(4,1,2); 

// Planes
filldraw(rp(vector(1.2,0,0))--rp(vector(1.2,0,1.2))--
         rp(vector(0,0,1.2))--rp(vector(0,0,0))--cycle,
         gray(.9), gray(.9));
filldraw(rp(vector(0,1.2,0))--rp(vector(0,1.2,1.2))--
         rp(vector(0,0,1.2))--rp(vector(0,0,0))--cycle,
         gray(.9), gray(.9));
filldraw(rp(vector(1.2,0,0))--rp(vector(1.2,1.2,0))--
         rp(vector(0,1.2,0))--rp(vector(0,0,0))--cycle,
         gray(0.9), gray(.9));

{ // Fixed axes
  pen hou = currentpen;
  
  currentpen = rgb(0,0.7,0);
  cartaxes(1.5,1.5,1.5);  
  currentpen = hou;
  label("$\rm O$", rp(vector(0,0,0)),W);
}

// Point Q
pO = point_on_sphere(vector(0,0,0), radius, 90, aux);
pB = point_on_sphere(vector(0,0,0), radius, lambda, aux);

draw(rp(vector(0,0,0))--rp(pO),dashed);
draw("$\vec{R}$",rp(vector(0,0,0))--rp(pB),Arrow);
label("$\rm Q$", rp(pB),N+3*W);
angline(vector(0,0,pB.z), pB, vector(0,0,0), 0.15, "$\lambda$", N
+0.2*E);

// Particle
vector m = pB-vector(0.26,-0.4,0.28);

real width=5;
labeldot("$m$",rp(m),SE,linewidth(width));
draw("$\vec{\rho}$",rp(vector(0,0,0))--rp(m),Arrow,PenMargin(0,width));
draw("$\bm{r}$",rp(pB)--rp(m),Arrow,PenMargin(0,width));

{ // Sphere
  real sirk_rad;
  sirk_rad = sqrt(pB.x^2+pB.y^2);
  
  draw(circseg(vector(0,0,pB.z), sirk_rad, (90,90), (0,90)),dashed);
  draw(circseg(vector(0,0,0), radius, (0,90), (aux,aux)),dashed);
  draw(circseg(vector(0,0,0), radius, (0,90), (0,0)),thickp);
  draw(circseg(vector(0,0,0), radius, (0,90), (90,90)),thickp);
  draw(circseg(vector(0,0,0), radius, (90,90), (0,90)),thickp);
}

{ // Moving axes
  vector i,j,k;

  i = unit(point_on_sphere(vector(0,0,0),1,90+lambda,aux));
  k = unit(pB);
  j = ncrossprod(k,i);
  draw("$x$",rp(pB)--rp(pB+0.2*i),1.0,2W,red,Arrow);
  draw("$y$",rp(pB)--rp(pB+0.32*j),1.0,red,Arrow);
  draw("$z$",rp(pB)--rp(pB+0.26*k),1.0,red,Arrow);
}

draw("$\omega\vec{K}$",circseg(vector(0,0,0.9), 0.2, (90,90),
			       (-120,160)),N,Arrow);

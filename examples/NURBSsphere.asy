import three;

/* Reference:
@article{Qin97,
  title={{Representing quadric surfaces using NURBS surfaces}},
  author={Qin, K.},
  journal={Journal of Computer Science and Technology},
  volume={12},
  number={3},
  pages={210--216},
  year={1997},
  publisher={Springer}
}
*/

size(10cm);
currentprojection=perspective(5,4,2,autoadjust=false);

// udegree=2, vdegree=3, nu=3, nv=4;

real[] W={2/3,1/3,1};
real[] w={1,1/3,1/3,1};

// 10 distinct control points
triple[][] P={{(0,0,1),(-2,-2,1),(-2,-2,-1),(0,0,-1)},
              {(0,0,1),(2,-2,1),(2,-2,-1),(0,0,-1)},
              {(0,0,1),(2,2,1),(2,2,-1),(0,0,-1)},
              {(0,0,1),(-2,2,1),(-2,2,-1),(0,0,-1)}};

P.cyclic=true;

real[][] weights=new real[3][4];
for(int i=0; i < 3; ++i)
for(int j=0; j < 4; ++j)
  weights[i][j]=W[i]*w[j];

real[] uknot={0,0,1/3,1/2,1,1};
real[] vknot={0,0,0,0,1,1,1,1};

int N=1;

for(int k=0; k < N; ++k)
for(int i=0; i < 4; ++i)
  draw(shift(k*Z)*P[i:i+3],uknot,vknot,weights,blue);

// draw(unitsphere,red+opacity(0.1));

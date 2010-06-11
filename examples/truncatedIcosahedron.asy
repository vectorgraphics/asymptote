import graph3;

size(200);

defaultrender.merge=true;

real c=(1+sqrt(5))/2;

triple[] z={(c,1,0),(-c,1,0),(-c,-1,0),(c,-1,0)};
triple[] x={(0,c,1),(0,-c,1),(0,-c,-1),(0,c,-1)};
triple[] y={(1,0,c),(1,0,-c),(-1,0,-c),(-1,0,c)};

triple[][] Q={
  {(c,1,0),(1,0,-c),(0,c,-1),(0,c,1),(1,0,c),(c,-1,0)},
  {(-c,1,0),(0,c,1),(0,c,-1),(-1,0,-c),(-c,-1,0),(-1,0,c)},
  {(-c,-1,0),(-c,1,0),(-1,0,-c),(0,-c,-1),(0,-c,1),(-1,0,c)},
  {(c,-1,0),(c,1,0),(1,0,c),(0,-c,1),(0,-c,-1),(1,0,-c)},
  {(0,c,1),(0,c,-1),(-c,1,0),(-1,0,c),(1,0,c),(c,1,0)},
  {(0,-c,1),(0,-c,-1),(-c,-1,0),(-1,0,c),(1,0,c),(c,-1,0)},
  {(0,-c,-1),(0,-c,1),(c,-1,0),(1,0,-c),(-1,0,-c),(-c,-1,0)},
  {(0,c,-1),(0,c,1),(c,1,0),(1,0,-c),(-1,0,-c),(-c,1,0)},
  {(1,0,c),(-1,0,c),(0,-c,1),(c,-1,0),(c,1,0),(0,c,1)},
  {(1,0,-c),(-1,0,-c),(0,-c,-1),(c,-1,0),(c,1,0),(0,c,-1)},
  {(-1,0,-c),(1,0,-c),(0,c,-1),(-c,1,0),(-c,-1,0),(0,-c,-1)},
  {(-1,0,c),(1,0,c),(0,c,1),(-c,1,0),(-c,-1,0),(0,-c,1)}
};

real R=abs(interp(Q[0][0],Q[0][1],1/3));

triple[][] P;
for(int i=0; i < Q.length; ++i) {
  P[i]=new triple[] ;
  for(int j=0; j < Q[i].length; ++j) {
    P[i][j]=Q[i][j]/R;
  }
}

for(int i=0; i < P.length; ++i) {
  for(int j=1; j < P[i].length; ++j) {
    triple C=P[i][0];
    triple A=P[i][j];
    triple B=P[i][j % 5+1];
    triple[] sixout=new
      triple[] {interp(C,A,1/3),interp(C,A,2/3),interp(A,B,1/3),interp(A,B,2/3),
                interp(B,C,1/3),interp(B,C,2/3)};
    triple M=(sum(sixout))/6;
    triple[] sixin=sequence(new triple(int k) {
        return interp(sixout[k],M,0.1);
      },6);
    draw(surface(reverse(operator--(...sixout)--cycle)^^
                 operator--(...sixin)--cycle,planar=true),magenta);
  }
}

for(int i=0; i < P.length; ++i) {
  triple[] fiveout=sequence(new triple(int k) {
      return interp(P[i][0],P[i][k+1],1/3);
    },5);
  triple M=(sum(fiveout))/5;
  triple[] fivein=sequence(new triple(int k) {
      return interp(fiveout[k],M,0.1);
    },5);
  draw(surface(reverse(operator--(...fiveout)--cycle)^^
               operator--(...fivein)--cycle,planar=true),cyan);
}




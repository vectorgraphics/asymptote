import three;
//currentprojection=perspective(-2,5,1);
currentprojection=orthographic(-2,5,1);

size(10cm);

surface s=surface((0,0,0)--(3,0,0)--(1.5,3*sqrt(3)/2,0)--cycle,
                  new triple[] {(1.5,sqrt(3)/2,2)});

draw(s,material(red,shininess=0.85));

/*
int n=10;
for(int i=0; i < n; ++i) {
  real u=i/n;
  for(int j=0; j < n-i; ++j) {
    real v=j/n;
    real w=1-u-v;
    //    u=0;v=1;w=0;
    //    write(u,v,w);
    triple z=s.s[0].pointtriangular(u,v);
    triple n=s.s[0].normal(u,v);
    draw(z--z+0.5*unit(n),red,Arrow3);
    dot(s.s[0].P[3][3]);
  }
}
*/

//dot(s.s[0].pointtriangular(1/3,1/3));

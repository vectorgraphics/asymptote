import three;

size(100,100);
currentprojection=perspective(10,7,40);

int N=4;
real f=1+1/N;

for(int k=0; k < N; ++k) {
  for(int m=0; m < N; ++m) {
    for(int n=0; n < N; ++n) {
      draw(shift((n,m,k)*f)*unitcube,red+opacity(0.5));
    }
  }
}

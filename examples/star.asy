size(100);
import math;

int n=5;
path p;

int i=0;
do {
  p=p--unityroot(n,i);
  i=(i+2) % n;
} while(i != 0);

filldraw(p--cycle,red+evenodd);

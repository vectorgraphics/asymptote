import TestLib;
import math;

StartTest("determinant");
assert(determinant(new real[][] {{1}}) == 1);
assert(determinant(new real[][] {{1,2},{3,4}}) == -2);
real e=1e-20;
assert(close(determinant(new real[][] {{1e,2e},{3e,4e}}),-2e-40));
assert(close(determinant(new real[][] {{1,2,3},{4,5,6},{7,8,9}}),0));
assert(close(determinant(new real[][] {{1,2,3,4},
			     {5,6,7,8},{9,10,11,12},{13,14,15,16}}),0));
assert(close(determinant(new real[][] {{1,2,3,4},
			     {5,0,7,8},{9,10,0,12},{13,14,15,16}}),-2376));
EndTest();

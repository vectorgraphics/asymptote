import TestLib;
import math;

StartTest("determinant");
assert(determinant(new real[][] {{0}}) == 0);
assert(determinant(new real[][] {{1}}) == 1);
assert(determinant(new real[][] {{1,2},{3,4}}) == -2);
real e=1e-20;
assert(close(determinant(new real[][] {{1e,2e},{3e,4e}}),-2e-40));
assert(close(determinant(new real[][] {{1,2,3},{4,5,6},{7,8,9}}),0));
assert(close(determinant(new real[][] {{1,2},{1,2}}),0));
assert(close(determinant(new real[][] {{1,2,3,4},
			     {5,6,7,8},{9,10,11,12},{13,14,15,16}}),0));
assert(close(determinant(new real[][] {{1,2,3,4},
			     {5,0,7,8},{9,10,0,12},{13,14,15,16}}),-2376));
assert(close(determinant(new real[][]{{1,-2,3,0},{4,-5,6,2},{-7,-8,10,5},
				      {1,50,1,-2}}),-4588));
EndTest();

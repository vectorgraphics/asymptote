size(0,100);
import math; 
import stats; 
 
currentpen=magenta; 
// A centered random number 
real crand() {return unitrand()*5;} 
 
real r1; 
pair pcenter; 
 
for(int i=0; i < 6000; ++i) { 
 
r1 = unitrand()/10; 
pcenter = ( crand(), crand()); 
Draw(circle(pcenter,r1)); 
} 

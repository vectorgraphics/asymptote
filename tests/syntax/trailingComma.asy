import TestLib;
// Note: If there's actually a syntax error, `StartTest` will never
// get called. However, in the case of a successful test, it does
// provide a record that the test was run and succeeded.
StartTest('Trailing Commas');
void f(pair a, ...string[] b) {}
f((1,2), ...new string[] { 'a', 'b' });

// The following lines represent syntax that could feasibly be allowed
// (see the `oxfordComma` branch) but currently is not. Some of these
// likely will be allowed at some point in the future; others will
// remain here forever as examples of things that are possible but
// undesirable.
//
//access three, trembling,;
//from three unravel surface, shift,;
//from three access surface, shift,;
//typedef import(T,);
//access mapArray(Src=string, Dst=int,) as mapStringToInt;
//from mapArray(Src=string, Dst=int,) access map;
//from mapArray(Src=string, Dst=int) access map,;
//int a=1, b=2,;
//for (int i=0, j=7,; i < 10; ++i) {}
//int f(int a, int b,);
//int f(int a, int b,) { return 0; }
//using T = int(string s,);
//f(3, 4,);
//(new int(int, int) { return 0; })(3,4,);
//new int(int,int,) { return 0; };
//new int[] (int,int,) { return new int[]; };
//pair p = (1, 2,);
//path p = (0,0) {1,1,} .. (1,0);
//import three; path3 p = (0,0,0) {1,1,1,} .. (1,0,0);

EndTest();

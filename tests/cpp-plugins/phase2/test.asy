access "phase2" as p;

// callable<int(int)>
write(p.apply_int(new int(int x) { return x * x; }, 7));   // 49

// callable<string(string)>
write(p.apply_str(new string(string s) { return s + "!"; }, "hello"));  // hello!

// callable + accumulator
write(p.sum_apply(new int(int i) { return i + 1; }, 5));   // 1+2+3+4+5 = 15

// result<int>
var r1 = p.safe_divide(10, 3);
write(r1.found);    // true
write(r1.value);    // 3

var r2 = p.safe_divide(10, 0);
write(r2.found);    // false

// result<string>
var r3 = p.lookup_name(2);
write(r3.found);    // true
write(r3.value);    // bob

var r4 = p.lookup_name(99);
write(r4.found);    // false

// callable + result
var r5 = p.find_first(new bool(int i) { return i > 3; }, 10);
write(r5.found);    // true
write(r5.value);    // 4

var r6 = p.find_first(new bool(int i) { return i > 100; }, 10);
write(r6.found);    // false

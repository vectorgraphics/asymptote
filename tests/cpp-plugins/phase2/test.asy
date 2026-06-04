access "phase2" as p;

// callable<int(int)>
assert(p.apply_int(new int(int x) { return x * x; }, 7) == 49);

// callable<string(string)>
assert(p.apply_str(new string(string s) { return s + "!"; }, "hello") == "hello!");

// callable + accumulator
assert(p.sum_apply(new int(int i) { return i + 1; }, 5) == 15);  // 1+2+3+4+5

// result<int>
var r1 = p.safe_divide(10, 3);
assert(r1.found);
assert(r1.value == 3);

var r2 = p.safe_divide(10, 0);
assert(!r2.found);

// result<string>
var r3 = p.lookup_name(2);
assert(r3.found);
assert(r3.value == 'bob');

var r4 = p.lookup_name(99);
assert(!r4.found);

// callable + result
var r5 = p.find_first(new bool(int i) { return i > 3; }, 10);
assert(r5.found);
assert(r5.value == 4);

var r6 = p.find_first(new bool(int i) { return i > 100; }, 10);
assert(!r6.found);

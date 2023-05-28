import TestLib;

StartTest("csv");
{
    real[] a=input("io/input_with_nan.csv").csv();
    assert(a.length == 20);
    for (int i=0; i<4; ++i)
	assert(a[i] == i);
    for (int i=4; i<8; ++i)
	assert(isnan(a[i]));
    for (int i=8; i<12; ++i)
	assert(isnan(a[i]));
    for (int i=12; i<16; ++i)
	assert(a[i] == inf);
    for (int i=16; i<20; ++i)
	assert(a[i] == -inf);
}
EndTest();

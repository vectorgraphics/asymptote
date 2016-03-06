import TestLib;

StartTest("Ghostscript");
bool uptodate=texpath("A").length != 0;
if(!uptodate) {
  write();
  write("Incompatible Ghostscript version!");
  write("Ghostscript version <= 9.13: please recompile with CFLAGS=-DEPSWRITE");
  write("Ghostscript version >= 9.14: please recompile without CFLAGS=-DEPSWRITE");
  write();
}
assert(uptodate);
EndTest();

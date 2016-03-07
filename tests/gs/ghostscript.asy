import TestLib;

StartTest("Ghostscript");
bool uptodate=texpath("A").length != 0;
if(!uptodate) {
  write();
  write();
  write("Incompatible Ghostscript version!");
  write("Please set environment variable ASYMPTOTE_EPSDRIVER to");
  write("\"epswrite\" for Ghostscript < 9.14 and to \"eps2write\" for Ghostscript >= 9.14");
  write();
}
assert(uptodate);
EndTest();

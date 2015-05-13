import TestLib;

StartTest("Ghostscript");
bool uptodate=texpath("A").length != 0;
if(!uptodate) {
  write();
  write();
  write("Please install Ghostscript version 9.14 or later.");
  write();
}
assert(uptodate);
EndTest();

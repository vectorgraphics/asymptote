access settings;

// Return code: 0=none, 1=step, 2=next.

int debugger(string file, int line, int column) {
  static int saveverbose=settings.verbose;
  settings.verbose=saveverbose;
  static bool debugging=true;
  static bool first=true;
  if(debugging) {
    while(true) {
      string prompt=file+": "+(string) line+"."+(string) column;
      if(first) {prompt += " (h for help)"; first=false;}
      prompt += "? [%s] ";
      string s=getstring(name="debug",prompt=prompt);
      if(s == "h") {write("c:continue n:next s:step t:trace q:quit"); continue;}
      if(s == "c") break;
      if(s == "s") return 1;
      if(s == "n") return 2;
      if(s == "q") {debugging=false; break;}
      if(s == "t") {settings.verbose=5; break;}
      _eval(s+";",true);
    }
  }
  return 0;
}

atbreakpoint(debugger);

int debuggerlines=5;

void stop(string file, string text) 
{
  string[] source=input(file);
  for(int line=0; line < source.length; ++line)
    if(find(source[line],text) >= 0) {
      stop(file,line+1);
      return;
    }
  write("no matching line in "+file+": \""+text+"\"");
}

int debugger(string file, int line, int column) 
{
  static int saveverbose=settings.verbose;
  settings.verbose=saveverbose;
  static bool debugging=true;
  if(debugging) {
    static string lastfile;
    static string[] source;
    bool help=false;
    while(true) {
      if(file != lastfile) {source=input(file); lastfile=file;}
      write();
      for(int i=max(line-debuggerlines,0); i < line; ++i)
	write(source[i]);
      for(int i=0; i < column-1; ++i)
	write(" ",none);
      write("^");
      if(help) {
	write("c:continue f:file h:help l:line n:next r:return s:step t:trace q:quit e:exit");
	help=false;
      }
      string prompt=file+": "+(string) line+"."+(string) column;
      prompt += "? [%s] ";
      string s=getstring(name="debug",default="h",prompt=prompt,save=false);
      if(s == "h") {help=true; continue;}
      if(s == "c") break;
      if(s == "s") return 1; // step
      if(s == "n") return 2; // next
      if(s == "l") return 3; // line
      if(s == "f") return 4; // file
      if(s == "r") return 5; // return
      if(s == "t") {settings.verbose=5; break;} // trace
      if(s == "q") abort(); // quit
      if(s == "x") {debugging=false; break;} // exit
      _eval(s+";",true);
    }
  }
  return 0;
}

atbreakpoint(debugger);

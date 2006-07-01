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

string debugger(string file, int line, int column) 
{
  int verbose=settings.verbose;
  settings.verbose=0;
  static bool debugging=true;
  static string s;
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
      write("^"+(verbose == 5 ? " trace" : ""));

      if(help) {
	write("c:continue f:file h:help n:next r:return s:step t:trace q:quit e:exit");
	help=false;
      }

      string prompt=file+": "+(string) line+"."+(string) column;
      prompt += "? [%s] ";
      s=getstring(name="debug",default="h",prompt=prompt,save=false);
      if(s == "h") {help=true; continue;}
      if(s == "c" || s == "s" || s == "n" || s == "f" || s == "r")
	break;
      if(s == "q") abort(); // quit
      if(s == "x") {debugging=false; return "";} // exit
      if(s == "t") { // trace
	write(tab);
	if(verbose == 0) {
	  verbose=5;
	} else {
	  verbose=0;
	}
	continue;
      }
      _eval(s+";",true);
    }
  }
  settings.verbose=verbose;
  return s;
}

atbreakpoint(debugger);

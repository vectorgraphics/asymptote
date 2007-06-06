int debuggerlines=5;

int sourceline(string file, string text)
{
  string file=locatefile(file);
  string[] source=input(file);
  for(int line=0; line < source.length; ++line)
    if(find(source[line],text) >= 0) return line+1;
  write("no matching line in "+file+": \""+text+"\"");
  return 0;
}

void stop(string file, string text, code s=quote{})
{
  int line=sourceline(file,text);
  if(line > 0) stop(file,line,s);
}

void clear(string file, string text)
{
  int line=sourceline(file,text);
  if(line > 0) clear(file,line);
}

// Enable debugging.
bool debugging=true;

// Variables used by conditional expressions:
// e.g. stop("test",2,quote{ignore=(++count <= 10);});

bool ignore;
int count=0;

string debugger(string file, int line, int column, code s=quote{})
{
  int verbose=settings.verbose;
  settings.verbose=0;
  _eval(s,true);
  if(ignore) {
    ignore=false;
    settings.verbose=verbose;
    return "c";
  }
  static string s;
  if(debugging) {
    static string lastfile;
    static string[] source;
    bool help=false;
    while(true) {
      if(file != lastfile && file != "-") {source=input(file); lastfile=file;}
      write();
      for(int i=max(line-debuggerlines,0); i < min(line,source.length); ++i)
        write(source[i]);
      for(int i=0; i < column-1; ++i)
        write(" ",none);
      write("^"+(verbose == 5 ? " trace" : ""));

      if(help) {
        write("c:continue f:file h:help i:inst n:next r:return s:step t:trace q:quit x:exit");
        help=false;
      }

      string Prompt=file+": "+(string) line+"."+(string) column;
      Prompt += "? [%s] ";
      s=getstring(name="debug",default="h",prompt=Prompt,store=false);
      if(s == "h" || s == "?") {help=true; continue;}
      if(s == "c" || s == "s" || s == "n" || s == "i" || s == "f" || s == "r")
        break;
      if(s == "q") {debugging=false; abort();} // quit
      if(s == "x") {debugging=false; return "";} // exit
      if(s == "t") { // trace
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

string defaultformat(int n, string trailingzero="", bool fixed=false,
                     bool signed=true)
{
  return "$%"+trailingzero+"."+string(n)+(fixed ? "f" : "g")+"$";
}

string defaultformat=defaultformat(4);
string defaultseparator="\!\times\!";

string ask(string prompt)
{
  write(stdout,prompt);
  return stdin;
}

string getstring(string name="", string default="", string prompt="",
                 bool store=true)
{
  string[] history=history(name,1);
  if(history.length > 0) default=history[0];
  if(prompt == "") prompt=name+"? [%s] ";
  prompt=replace(prompt,new string[][] {{"%s",default}});
  string s=readline(prompt,name);
  if(s == "") s=default;
  else saveline(name,s,store);
  return s;
}

int getint(string name="", int default=0, string prompt="", bool store=true)
{
  return (int) getstring(name,(string) default,prompt,store);
}

real getreal(string name="", real default=0, string prompt="", bool store=true)
{
  return (real) getstring(name,(string) default,prompt,store);
}

pair getpair(string name="", pair default=0, string prompt="", bool store=true)
{
  return (pair) getstring(name,(string) default,prompt,store);
}

triple gettriple(string name="", triple default=(0,0,0), string prompt="",
		 bool store=true)
{
  return (triple) getstring(name,(string) default,prompt,store);
}

// returns a string with all occurrences of string 'before' in string 's'
// changed to string 'after'.
string replace(string s, string before, string after) 
{
  return replace(s,new string[][] {{before,after}});
}

// Like texify but don't convert embedded TeX commands: \${}
string TeXify(string s) 
{
  static string[][] t={{"&","\&"},{"%","\%"},{"_","\_"},{"#","\#"},{"<","$<$"},
                       {">","$>$"},{"|","$|$"},{"^","$\hat{\ }$"},
                       {"~","$\tilde{\ }$"},{" ","\phantom{ }"}};
  return replace(s,t);
}

private string[][] trans1={{'\\',"\backslash "},
                           {"$","\$"},{"{","\{"},{"}","\}"}};
private string[][] trans2={{"\backslash ","$\backslash$"}};

// Convert string to TeX
string texify(string s) 
{
  return TeXify(replace(replace(s,trans1),trans2));
}

// Convert string to TeX, preserving newlines
string verbatim(string s)
{
  bool space=substr(s,0,1) == '\n';
  static string[][] t={{'\n',"\\"}};
  t.append(trans1);
  s=TeXify(replace(replace(s,t),trans2));
  return space ? "\ "+s : s;
}

// Split a string into an array of substrings delimited by delimiter
// If delimiter is an empty string, use space delimiter but discard empty
// substrings. TODO: Move to C++ code.
string[] split(string s, string delimiter="")
{
  bool prune=false;
  if(delimiter == "") {
    prune=true;
    delimiter=" ";
  }
  
  string[] S;
  int last=0;
  int i;
  int N=length(delimiter);
  int n=length(s);
  while((i=find(s,delimiter,last)) >= 0) {
    if(i > last || (i == last && !prune))
      S.push(substr(s,last,i-last));
    last=i+N;
  }
  if(n > last || (n == last && !prune))
    S.push(substr(s,last,n-last));
  return S;
}

// Returns an array of strings obtained by splitting s into individual
// characters. TODO: Move to C++ code.
string[] array(string s)
{
  int len=length(s);
  string[] S=new string[len];
  for(int i=0; i < len; ++i)
    S[i]=substr(s,i,1);
  return S;
}

// Concatenate an array of strings into a single string.
// TODO: Move to C++ code.
string operator +(...string[] a)
{
  string S;
  for(string s : a)
    S += s;
  return S;
}

int system(string s) 
{
  return system(split(s));
}

int[] operator ecast(string[] a)
{
  return sequence(new int(int i) {return (int) a[i];},a.length);
}

real[] operator ecast(string[] a)
{
  return sequence(new real(int i) {return (real) a[i];},a.length);
}

// Read contents of file as a string.
string file(string s)
{
  file f=input(s);
  string s;
  while(!eof(f)) {
    s += f+'\n';
  }
  return s;
}

string italic(string s)
{
  return s != "" ? "{\it "+s+"}" : s;
}

string baseline(string s, string template="\strut") 
{ 
  return s != "" && settings.tex != "none" ? "\vphantom{"+template+"}"+s : s;
}

string math(string s)
{
  return s != "" ? "$"+s+"$" : s;
}

private void notimplemented(string text) 
{
  abort(text+" is not implemented for the '"+settings.tex+"' TeX engine");
}

string jobname(string name)
{
  int pos=rfind(name,"-");
  return pos >= 0 ? "\ASYprefix\jobname"+substr(name,pos) : name;
}

string graphic(string name, string options="")
{
  if(latex()) {
    if(options != "") options="["+options+"]";
    string includegraphics="\includegraphics"+options;
    return includegraphics+"{"+(settings.inlinetex ? jobname(name) : name)+"}";
  }
  if(settings.tex != "context")
    notimplemented("graphic");
  return "\externalfigure["+name+"]["+options+"]";
}

string graphicscale(real x)
{
  return string(settings.tex == "context" ? 1000*x : x);
}

string minipage(string s, real width=100bp)
{
  if(latex())
    return "\begin{minipage}{"+(string) (width/pt)+"pt}"+s+"\end{minipage}";
  if(settings.tex != "context")
    notimplemented("minipage");
  return "\startframedtext[none][frame=off,width="+(string) (width/pt)+
    "pt]"+s+"\stopframedtext";
}

void usepackage(string s, string options="")
{
  if(!latex()) notimplemented("usepackage");
  string usepackage="\usepackage";
  if(options != "") usepackage += "["+options+"]";
  texpreamble(usepackage+"{"+s+"}");
}

void pause(string w="Hit enter to continue") 
{
  write(w);
  w=stdin;
}

string format(string format=defaultformat, bool forcemath=false, real x,
              string locale="")
{
  return format(format,forcemath,defaultseparator,x,locale);
}

string phantom(string s)
{
  return settings.tex != "none" ? "\phantom{"+s+"}" : "";
}

string[] spinner=new string[] {'|','/','-','\\'};
spinner.cyclic=true;

void progress(bool3 init=default)
{
  static int count=-1;
  static int lastseconds=-1;
  if(init == true) {
    lastseconds=0;
    write(stdout,' ',flush);
  } else
    if(init == default) {
    int seconds=seconds();
    if(seconds > lastseconds) {
      lastseconds=seconds;
      write(stdout,'\b'+spinner[++count],flush);
    }
  } else
      write(stdout,'\b',flush);
}

restricted int ocgindex=0;

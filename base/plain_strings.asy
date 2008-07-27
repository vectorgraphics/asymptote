string defaultformat(int n, bool trailingzero=false, bool fixed=false)
{
  return "$%"+(trailingzero ? "#" : "")+"."+string(n)+(fixed ? "f" : "g")+"$";
}

string defaultformat=defaultformat(4);

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
string[] split(string s, string delimiter)
{
  string[] S;
  int last=0;
  int i;
  int N=length(delimiter);
  while((i=find(s,delimiter,last)) >= 0) {
    if(i > last)
      S.push(substr(s,last,i-last));
    last=i+N;
  }
  int n=length(s);
  if(n > last)
    S.push(substr(s,last,n-last));
  return S;
}

real[] operator ecast(string[] a)
{
  real[] b=new real[a.length];
  for(int i=0; i < a.length; ++i)
    b[i]=(real) a[i];
  return b;
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
  if(s == "") return s;
  return "{\it "+s+"}";
}

string baseline(string s, align align=S, string template="M") 
{
  if(s == "") return s;
  return align.dir.y <= -0.5*abs(align.dir.x) ? 
    "\ASYbase{"+template+"}{"+s+"}" : s;
}

string math(string s)
{
  if(s == "") return s;
  return "$"+s+"$";
}

string graphic(string name, string options="")
{
  if(options != "") options="["+options+"]";
  return "\includegraphics"+options+"{"+name+"}";
}

string minipage(string s, real width=100bp)
{
  if(latex())
    return "\begin{minipage}{"+(string) (width*pt)+"pt}"+s+"\end{minipage}";
  write("warning: minipage requires -tex latex or -tex pdflatex");
  return "";
}

void usepackage(string s, string options="")
{
  if(latex()) {
    string usepackage="\usepackage";
    if(options != "") usepackage += "["+options+"]";
    texpreamble(usepackage+"{"+s+"}");
    return;
  }
  write("warning: usepackage requires -tex latex or -tex pdflatex");
}

void pause(string w="Hit enter to continue") 
{
  write(w);
  w=stdin;
}

string math(real x)
{
  return math((string) x);
}

string format(real x)
{
  return format(defaultformat,x);
}

string phantom(string s)
{
  return "\phantom{"+s+"}";
}

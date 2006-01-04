string ask(string prompt)
{
  write(stdout,prompt);
  return stdin;
}

string getstring(string name="", string default="", string prompt="",
		 bool save=true)
{
  return readline(prompt == "" ? name+"? [%s] " : prompt,
		  save ? name : '\012'+name,default);
}

int getint(string name="", int default=0, string prompt="", bool save=true)
{
  return (int) getstring(name,(string) default,prompt,save);
}

real getreal(string name="", real default=0, string prompt="", bool save=true)
{
  return (real) getstring(name,(string) default,prompt,save);
}

pair getpair(string name="", pair default=0, string prompt="", bool save=true)
{
  return (pair) getstring(name,(string) default,prompt,save);
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
		       {">","$>$"},{"|","$|$"},{"^","$\hat{\ }$"},{". ",".\ "},
                       {"~","$\tilde{\ }$"}};
  return replace(s,t);
}

// Convert string to TeX
string texify(string s) 
{
  static string[][] t={{'\\',"\backslash "},{"$","\$"},{"{","\{"},{"}","\}"}};
  static string[][] u={{"\backslash ","$\backslash$"}};
  return TeXify(replace(replace(s,t),u));
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

string includegraphics(string name, string options="")
{
  if(options != "") options="["+options+"]";
  return "\includegraphics"+options+"{"+name+"}";
}

string minipage(string s, real width=100pt)
{
  return "\begin{minipage}{"+(string) (width*pt)+"pt}"+s+"\end{minipage}";
}

void pause(string w="Hit enter to continue") 
{
  write(w);
  w=stdin;
}

void newpage() 
{
  tex("\newpage");
  layer();
}

string math(real x)
{
  return math((string) x);
}

string format(real x)
{
  return format(defaultformat,x);
}


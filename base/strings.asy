string ask(string prompt)
{
  write(stdout,prompt);
  return stdin;
}

public string getstringprefix=".asy_";

void savestring(string name, string value, string prefix=getstringprefix)
{
  file out=output(prefix+name);
  write(out,value);
  close(out);
}

string getstring(string name, string default="", string prompt="",
		 string prefix=getstringprefix, bool save=true)
{
  string value;
  file in=input(prefix+name,false);
  if(error(in)) value=default;
  else value=in;
  if(prompt == "") prompt=name+"? ["+value+"] ";
  string input=ask(prompt);
  if(input != "") value=input;
  if(save) savestring(name,value);
  return value;
}

real getreal(string name, real default=0, string prompt="",
	     string prefix=getstringprefix, bool save=true)
{
  string value=getstring(name,(string) default,prompt,getstringprefix,false);
  real x=(real) value;
  if(save) savestring(name,value);
  return x;
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

